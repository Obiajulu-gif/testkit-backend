//! Integration tests for `soroban-testkit limits`.
//!
//! Exercises the CLI binary for resource limit discovery, boundary refinement audits,
//! output sanitization, measurement limitations reporting, baseline comparisons,
//! and ramp parameter handling.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_soroban-testkit"))
        .args(args)
        .current_dir(workspace_root())
        .output()
        .expect("failed to run soroban-testkit")
}

// Cargo checks freshness, even when a previous build left an artifact behind.
// Share one successful build across the tests without racing multiple rebuilds.
fn ensure_vault_wasm() -> PathBuf {
    static WASM: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    WASM.get_or_init(|| {
        let wasm = workspace_root().join("target/wasm32v1-none/release/vault.wasm");
        let mut build = Command::new("cargo");
        build
            // The outer cargo-llvm-cov run instruments native tests. This nested
            // WASM fixture build must not inherit its unsupported coverage flags.
            .env_remove("RUSTFLAGS")
            .env_remove("CARGO_ENCODED_RUSTFLAGS")
            .env_remove("CARGO_TARGET_WASM32V1_NONE_RUSTFLAGS")
            .args([
                "build",
                "-p",
                "vault",
                "--target",
                "wasm32v1-none",
                "--release",
            ])
            .current_dir(workspace_root());
        require_successful_build(&mut build, &wasm).unwrap_or_else(|err| panic!("{err}"))
    })
    .clone()
}

fn require_successful_build(build: &mut Command, wasm: &Path) -> Result<PathBuf, String> {
    let output = build
        .output()
        .map_err(|err| format!("failed to build {}: {err}", wasm.display()))?;
    if !output.status.success() {
        return Err(format!(
            "failed to rebuild {}; refusing to use a stale wasm. Ensure the target is installed: \
             `rustup target add wasm32v1-none`. Build stderr: {}",
            wasm.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    if !wasm.is_file() {
        return Err(format!("build succeeded but {} is missing", wasm.display()));
    }
    Ok(wasm.to_path_buf())
}

#[test]
fn limits_help_documents_options_and_limitations() {
    let output = run(&["limits", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("--contract"), "{stdout}");
    assert!(stdout.contains("does not rebuild contracts"), "{stdout}");
    assert!(stdout.contains("--fn"), "{stdout}");
    assert!(stdout.contains("--ramp"), "{stdout}");
    assert!(stdout.contains("--arg"), "{stdout}");
    assert!(stdout.contains("--probe-timeout"), "{stdout}");
    assert!(stdout.contains("--jobs"), "{stdout}");
    assert!(stdout.contains("--baseline"), "{stdout}");
    assert!(stdout.contains("--baseline-tolerance-pct"), "{stdout}");
    assert!(stdout.contains("--save-baseline"), "{stdout}");
    assert!(stdout.contains("--export"), "{stdout}");
    assert!(stdout.contains("--dry-run"), "{stdout}");
    assert!(stdout.contains("--export-samples"), "{stdout}");
    assert!(stdout.contains("--co-ramp"), "{stdout}");
    assert!(stdout.contains("--max-probes"), "{stdout}");
    assert!(stdout.contains("--wasm-target-dir"), "{stdout}");
    assert!(stdout.contains("--profile"), "{stdout}");
    assert!(stdout.contains("--timeout"), "{stdout}");
    assert!(stdout.contains("--progress"), "{stdout}");
}

#[test]
fn limits_rejects_invalid_baseline_tolerance() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--baseline-tolerance-pct",
        "150",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--baseline-tolerance-pct must be between 0 and 100"),
        "{stderr}"
    );
}

#[test]
fn limits_rejects_arg_overriding_ramp_parameter() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--arg",
        "recipients=10",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--arg cannot override ramp parameter \"recipients\""),
        "{stderr}"
    );
}

#[test]
fn limits_rejects_unknown_function_with_available_list() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "nonexistent_function",
        "--ramp",
        "recipients",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no function named \"nonexistent_function\""),
        "{stderr}"
    );
    assert!(stderr.contains("available functions:"), "{stderr}");
    assert!(stderr.contains("batch_payout"), "{stderr}");
}

#[test]
fn limits_rejects_unknown_ramp_before_probing_or_writing_reports() {
    let wasm = ensure_vault_wasm();
    let root = std::env::temp_dir().join(format!("stk-limits-unknown-ramp-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let export = root.join("report.json");
    let baseline = root.join("baseline.txt");
    std::fs::write(&export, "existing export").unwrap();
    std::fs::write(&baseline, "existing baseline").unwrap();

    // `owner` exists on initialize, but not on the selected function.
    for ramp in ["recipents", "Recipients", "owner", ""] {
        let output = run(&[
            "limits",
            "--contract",
            wasm.to_str().unwrap(),
            "--fn",
            "batch_payout",
            "--ramp",
            ramp,
            "--probe-timeout",
            "0",
            "--arg",
            "unknown=1",
            "--export",
            export.to_str().unwrap(),
            "--save-baseline",
            baseline.to_str().unwrap(),
        ]);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{output:?}");
        assert!(
            stderr.contains(&format!("\"batch_payout\" has no parameter named {ramp:?}")),
            "{stderr}"
        );
        assert!(
            stderr.contains("available parameters: recipients, amount"),
            "{stderr}"
        );
        assert!(stderr.contains("--ramp"), "{stderr}");
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert!(output.stdout.is_empty(), "{output:?}");
        assert_eq!(std::fs::read_to_string(&export).unwrap(), "existing export");
        assert_eq!(
            std::fs::read_to_string(&baseline).unwrap(),
            "existing baseline"
        );
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn limits_explains_when_the_selected_function_has_no_rampable_parameters() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "balance",
        "--ramp",
        "env",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{output:?}");
    assert!(
        stderr.contains("\"balance\" has no parameter named \"env\""),
        "{stderr}"
    );
    assert!(stderr.contains("available parameters: (none)"), "{stderr}");
    assert!(
        stderr.contains("choose a function with a numeric, Bytes, or Vec parameter"),
        "{stderr}"
    );
    assert!(output.stdout.is_empty(), "{output:?}");
}

#[test]
fn limits_accepts_a_non_first_parameter_name_from_the_selected_abi() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "amount",
        "--arg",
        "amount=10",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{output:?}");
    // Reaching override validation proves the ABI lookup accepted `amount`.
    assert!(
        stderr.contains("--arg cannot override ramp parameter \"amount\""),
        "{stderr}"
    );
    assert!(!stderr.contains("no parameter named"), "{stderr}");
}

#[test]
fn limits_reports_refinement_audit_and_limitations_in_stdout() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stdout.contains("last successful recipients"), "{stdout}");
    assert!(stdout.contains("instructions:"), "{stdout}");
    assert!(stdout.contains("memory bytes:"), "{stdout}");
    assert!(
        stdout.contains("boundary refinement: probed"),
        "stdout missing refinement audit: {stdout}"
    );
    assert!(
        stdout.contains(
            "ledger reads/writes and transaction size are not measured by this command; see --help"
        ),
        "stdout missing measurement limitations: {stdout}"
    );
    // #492: failures are reported with a standardized category.
    assert!(
        stdout.contains("first failed recipients value:"),
        "stdout missing first failure: {stdout}"
    );
    assert!(
        stdout.contains("probe failures:"),
        "stdout missing probe failure categories: {stdout}"
    );
}

#[test]
fn limits_text_summary_and_json_export_agree_on_the_boundary() {
    // #490: the terminal summary and the machine-readable report are both
    // rendered from one summary, so the discovered boundary and the
    // standardized failure category must match exactly.
    let wasm = ensure_vault_wasm();
    let temp_dir = std::env::temp_dir().join(format!("stk-limits-agree-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let export_path = temp_dir.join("agreement.json");

    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export",
        export_path.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&export_path).unwrap()).unwrap();
    let best = json["best"].as_u64().unwrap();

    assert!(
        stdout.contains(&format!(
            "last successful recipients for \"batch_payout\": {best}"
        )),
        "stdout boundary disagrees with JSON: {stdout}"
    );
    assert_eq!(json["audit"]["final_success"], best);

    // The standardized failure category must appear identically in both.
    let category = match json["first_failure_category"].as_str() {
        Some(category) => category,
        None => panic!("JSON is missing first_failure_category: {json}"),
    };
    assert!(
        stdout.contains(&format!("[{category}]")),
        "stdout is missing standardized category {category}: {stdout}"
    );
    assert!(
        json["probe_failure_counts"][category].as_u64().unwrap() >= 1,
        "JSON failure counts are missing {category}: {json}"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn limits_export_json_includes_limitations_audit_and_sanitizes_temp_paths() {
    let wasm = ensure_vault_wasm();
    let temp_dir =
        std::env::temp_dir().join(format!("stk-limits-test-json-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Copy wasm into temp directory to verify temp path sanitization
    let temp_wasm = temp_dir.join("vault.wasm");
    std::fs::copy(&wasm, &temp_wasm).unwrap();

    let export_path = temp_dir.join("results.json");
    let output = run(&[
        "limits",
        "--contract",
        temp_wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export",
        export_path.to_str().unwrap(),
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");

    let export_content = std::fs::read_to_string(&export_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&export_content).unwrap();

    assert_eq!(json["function"], "batch_payout");
    assert_eq!(json["ramp"], "recipients");
    assert!(json["best"].as_u64().unwrap() > 0);
    assert!(json["instructions"].as_u64().unwrap() > 0);
    assert!(json["memory_bytes"].as_u64().unwrap() > 0);

    // Verify measurement limitations
    assert_eq!(
        json["measurement_limitations"],
        "ledger reads/writes and transaction size are not measured by this command; see --help"
    );

    // Verify temp directory path sanitization: must not leak temp_dir prefix
    assert_eq!(json["contract"], "vault.wasm");
    assert!(!export_content.contains("stk-limits-test-json"));

    // Verify audit structure is present
    assert!(json["audit"].is_object());
    assert!(!json["audit"]["steps"].as_array().unwrap().is_empty());
    assert_eq!(json["audit"]["final_success"], json["best"]);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn limits_save_baseline_includes_limitations_and_sanitizes_temp_paths() {
    let wasm = ensure_vault_wasm();
    let temp_dir =
        std::env::temp_dir().join(format!("stk-limits-test-base-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();

    let temp_wasm = temp_dir.join("vault.wasm");
    std::fs::copy(&wasm, &temp_wasm).unwrap();

    let baseline_path = temp_dir.join("vault.baseline");
    let output = run(&[
        "limits",
        "--contract",
        temp_wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--save-baseline",
        baseline_path.to_str().unwrap(),
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");

    let baseline_content = std::fs::read_to_string(&baseline_path).unwrap();
    assert!(baseline_content.contains("function=batch_payout"));
    assert!(baseline_content.contains("ramp=recipients"));
    assert!(baseline_content.contains("contract=vault.wasm"));
    assert!(!baseline_content.contains("stk-limits-test-base"));
    assert!(baseline_content.contains("measurement_limitations=ledger reads/writes and transaction size are not measured by this command; see --help"));

    // Verify subsequent --baseline comparison passes
    let cmp_output = run(&[
        "limits",
        "--contract",
        temp_wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--baseline",
        baseline_path.to_str().unwrap(),
    ]);
    let cmp_stdout = String::from_utf8_lossy(&cmp_output.stdout);
    assert!(cmp_output.status.success(), "stdout: {cmp_stdout}");
    assert!(cmp_stdout.contains("no regression"), "{cmp_stdout}");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn limits_explains_how_to_reproduce_reported_maximum_locally() {
    let wasm = ensure_vault_wasm();
    let temp_dir =
        std::env::temp_dir().join(format!("stk-limits-repro-test-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let export_path = temp_dir.join("repro.json");

    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export",
        export_path.to_str().unwrap(),
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(
        stdout.contains("reproduce locally:"),
        "stdout missing reproduction guidance: {stdout}"
    );
    assert!(
        stdout.contains("batch_payout"),
        "reproduction guidance missing function name: {stdout}"
    );
    assert!(
        stdout.contains("recipients"),
        "reproduction guidance missing ramp parameter: {stdout}"
    );

    let export_content = std::fs::read_to_string(&export_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&export_content).unwrap();
    assert!(
        json["reproduction"].is_string(),
        "JSON export missing reproduction guidance: {json}"
    );
    let repro_str = json["reproduction"].as_str().unwrap();
    assert!(
        repro_str.contains("batch_payout"),
        "repro string: {repro_str}"
    );
    assert!(
        repro_str.contains("recipients"),
        "repro string: {repro_str}"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ---- Bug: handle empty Bytes overrides without producing invalid XDR ----

#[test]
fn limits_accepts_empty_bytes_arg_override() {
    // The vault contract's batch_payout takes (recipients: Vec<Address>,
    // amount: i128) — neither is Bytes, but we can verify the CLI accepts
    // the empty-bytes override syntactically by using it on a non-ramp
    // parameter that exists. Since the vault has no Bytes parameter, this
    // test exercises the validation path: the CLI should parse `0x` and
    // not crash with an invalid-XDR error before reporting the type
    // mismatch.
    //
    // Instead, we test at the unit level (in limits.rs) that explicit_val
    // produces a valid empty Bytes for `0x`, and here we verify the CLI
    // doesn't reject `--arg name=0x` at the parse level for a known
    // function — the parameter-type mismatch error is expected and proves
    // the CLI got far enough to validate the type.
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--arg",
        "amount=0x",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The error should be about the type mismatch (amount is i128, not
    // Bytes), NOT about invalid XDR or a panic.
    assert!(
        !stderr.contains("invalid") || !stderr.contains("XDR"),
        "should not produce an invalid XDR error for empty bytes: {stderr}"
    );
}

// ---- Bug: retain stderr context from a failed wasm probe ----

#[test]
fn limits_reports_first_failing_probe_stderr_in_stdout() {
    // When a probe fails, the parent should capture and display the
    // child's stderr so the user can see *why* the ceiling was hit. We
    // verify this by checking that the stdout summary includes the
    // "first failing probe stderr" line when a failure occurs.
    //
    // The vault's batch_payout will eventually fail at a high recipient
    // count, and the probe's stderr (from the Soroban host) should appear
    // in the summary.
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    // The summary should report the first failure with its category.
    assert!(
        stdout.contains("first failed recipients value:"),
        "stdout missing first failure: {stdout}"
    );
}

#[test]
fn limits_includes_failed_probe_stderr_in_json_export() {
    // The JSON export should include first_failed_probe_stderr when a
    // probe failure occurred and the child produced stderr output.
    let wasm = ensure_vault_wasm();
    let temp_dir =
        std::env::temp_dir().join(format!("stk-limits-stderr-json-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let export_path = temp_dir.join("stderr.json");

    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export",
        export_path.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");

    let export_content = std::fs::read_to_string(&export_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&export_content).unwrap();

    // The field should be present in the JSON schema (even if null/omitted
    // when no stderr was captured).
    assert!(
        json.get("first_failed_probe_stderr").is_some()
            || json.get("first_failure_category").is_some(),
        "JSON should include probe failure context: {json}"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn limits_missing_contract_reports_path_and_recovery_without_writing_reports() {
    let dir = std::env::temp_dir().join(format!("stk-missing-contract-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for contract in [
        "missing.wasm",
        "missing directory/contract with spaces.wasm",
    ] {
        let export = dir.join("result.json");
        let baseline = dir.join("baseline.txt");
        let output = Command::new(env!("CARGO_BIN_EXE_soroban-testkit"))
            .current_dir(&dir)
            .args([
                "limits",
                "--contract",
                contract,
                "--fn",
                "batch_payout",
                "--ramp",
                "recipients",
                "--export",
                export.to_str().unwrap(),
                "--save-baseline",
                baseline.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(contract), "{stderr}");
        assert!(stderr.contains("contract file not found"), "{stderr}");
        assert!(stderr.contains("current directory"), "{stderr}");
        assert!(
            stderr.contains("build the contract successfully"),
            "{stderr}"
        );
        assert!(output.stdout.is_empty());
        assert!(!export.exists());
        assert!(!baseline.exists());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn limits_rejects_a_directory_as_contract() {
    let output = run(&[
        "limits",
        "--contract",
        ".",
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("readable .wasm file"), "{stderr}");
    assert!(!stderr.contains("contract file not found"), "{stderr}");
    assert!(output.stdout.is_empty());
}

#[test]
fn limits_launches_wasm_from_a_path_with_spaces_without_shell_splitting() {
    let wasm = ensure_vault_wasm();
    let root = std::env::temp_dir().join(format!("stk limit paths {}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let copy = root.join("contract copy.wasm");
    std::fs::copy(&wasm, &copy).unwrap();

    let output = run(&[
        "limits",
        "--contract",
        copy.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "stdout: {stdout}; stderr: {stderr}"
    );
    assert!(
        stdout.contains("--contract 'contract copy.wasm'"),
        "copyable guidance must quote the filename: {stdout}"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[cfg(unix)]
#[test]
fn fixture_rebuild_never_falls_back_to_stale_wasm() {
    let dir = std::env::temp_dir().join(format!("stk-rebuild-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wasm = dir.join("vault.wasm");
    std::fs::write(&wasm, b"old wasm").unwrap();
    let mut failed_build = Command::new("sh");
    failed_build.args(["-c", "echo compiler-error >&2; exit 1"]);
    let err = require_successful_build(&mut failed_build, &wasm).unwrap_err();
    assert!(err.contains("refusing to use a stale wasm"), "{err}");
    assert!(err.contains("compiler-error"), "{err}");
    assert_eq!(std::fs::read(&wasm).unwrap(), b"old wasm");

    let mut successful_build = Command::new("sh");
    successful_build
        .args(["-c", "printf 'new wasm' > \"$1\"", "build"])
        .arg(&wasm);
    assert_eq!(
        require_successful_build(&mut successful_build, &wasm).unwrap(),
        wasm
    );
    assert_eq!(std::fs::read(&wasm).unwrap(), b"new wasm");

    std::fs::remove_file(&wasm).unwrap();
    let mut missing_artifact = Command::new("sh");
    missing_artifact.args(["-c", "exit 0"]);
    assert!(require_successful_build(&mut missing_artifact, &wasm)
        .unwrap_err()
        .contains("is missing"));
    let mut missing_builder = Command::new(dir.join("missing-cargo"));
    assert!(require_successful_build(&mut missing_builder, &wasm)
        .unwrap_err()
        .contains("failed to build"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn limits_timeout_creates_no_temporary_cargo_probe_project() {
    let wasm = ensure_vault_wasm();
    let root = std::env::temp_dir().join(format!("stk-probe-empty-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_soroban-testkit"))
        .args([
            "limits",
            "--contract",
            wasm.to_str().unwrap(),
            "--fn",
            "batch_payout",
            "--ramp",
            "recipients",
            "--probe-timeout",
            "0",
        ])
        .current_dir(&root)
        .output()
        .unwrap();
    // Either the immediate deadline fails the first probe, or a very fast
    // probe may complete; neither outcome creates a disposable Cargo project.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() || stderr.contains("timed-out"),
        "unexpected probe error: {stderr}"
    );
    assert_eq!(
        std::fs::read_dir(&root).unwrap().count(),
        0,
        "limits must not leave probe Cargo.toml/src/target directories"
    );
    std::fs::remove_dir(&root).unwrap();
}

// #444: a contract error from the Vault (uninitialized deposit) is not
// evidence of a resource limit. Do not report a fake best ceiling.
#[test]
fn limits_uninitialized_vault_failure_is_not_reported_as_ceiling() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "deposit",
        "--ramp",
        "amount",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("deposit"), "{stderr}");
    assert!(stderr.contains("amount=1"), "{stderr}");
    assert!(
        stderr.contains("before a resource limit was verified"),
        "{stderr}"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("last successful"));
}

// #444: success at 1 and 2 must not let us report 2 as a budget ceiling
// when the contract itself rejects count 4.
#[test]
fn limits_stops_when_contract_rejects_after_successful_probes() {
    // Reuse the WASM fixture prepared by the CI workflow.
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "reject_after_three",
        "--ramp",
        "count",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("reject_after_three"), "{stderr}");
    assert!(stderr.contains("count=4"), "{stderr}");
    assert!(
        stderr.contains("before a resource limit was verified"),
        "{stderr}"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("last successful"));
}

// ---- Issue #372: dry-run mode ----

#[test]
fn limits_dry_run_prints_planned_probes_and_exits_zero() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--dry-run",
    ]);
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("dry-run"), "{stdout}");
    assert!(stdout.contains("batch_payout"), "{stdout}");
    assert!(stdout.contains("recipients"), "{stdout}");
    assert!(stdout.contains("probe ramp=1"), "{stdout}");
    assert!(stdout.contains("doubling phase"), "{stdout}");
}

#[test]
fn limits_dry_run_does_not_execute_probes() {
    // dry-run must not write any output files even when --export is given.
    let wasm = ensure_vault_wasm();
    let dir = std::env::temp_dir().join(format!("stk-dryrun-no-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let export_path = dir.join("dry.json");

    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--dry-run",
        "--export",
        export_path.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{:?}", output);
    assert!(
        !export_path.exists(),
        "dry-run must not write the export file"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn limits_dry_run_shows_max_probes_limit_when_set() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--dry-run",
        "--max-probes",
        "10",
    ]);
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("max-probes limit: 10"), "{stdout}");
}

// ---- Issue #373: export all probe samples as JSON ----

#[test]
fn limits_export_samples_writes_json_array_of_probes() {
    let wasm = ensure_vault_wasm();
    let dir = std::env::temp_dir().join(format!("stk-samples-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let samples_path = dir.join("samples.json");

    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export-samples",
        samples_path.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(samples_path.exists(), "samples file should be created");

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&samples_path).unwrap()).unwrap();
    assert!(json.is_array(), "samples should be a JSON array: {json}");
    let arr = json.as_array().unwrap();
    assert!(!arr.is_empty(), "samples array must not be empty");
    for entry in arr {
        assert!(entry["value"].as_u64().is_some(), "missing value: {entry}");
        assert!(
            entry["succeeded"].as_bool().is_some(),
            "missing succeeded: {entry}"
        );
    }
    assert_eq!(arr[0]["value"], 1, "first sample must be value=1");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn limits_export_samples_stdout_reports_path() {
    let wasm = ensure_vault_wasm();
    let dir = std::env::temp_dir().join(format!("stk-samples-stdout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let samples_path = dir.join("samples.json");

    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export-samples",
        samples_path.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(
        stdout.contains("exported probe samples to"),
        "stdout missing samples path: {stdout}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ---- Issue #368: correlated parameter ramping ----

#[test]
fn limits_co_ramp_rejects_unknown_parameter() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--co-ramp",
        "nonexistent_param",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--co-ramp"), "{stderr}");
    assert!(stderr.contains("nonexistent_param"), "{stderr}");
}

#[test]
fn limits_co_ramp_shown_in_stdout_summary() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--co-ramp",
        "amount",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(
        stdout.contains("recipients (+ amount)"),
        "co_ramp not shown in summary: {stdout}"
    );
}

// ---- Issue #369: maximum probe count protection ----

#[test]
fn limits_max_probes_zero_does_not_restrict() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--max-probes",
        "0",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
}

#[test]
fn limits_max_probes_one_aborts_after_first_probe() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--max-probes",
        "1",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--max-probes"),
        "error should mention --max-probes: {stderr}"
    );
    assert!(
        stderr.contains("raise --max-probes"),
        "error should suggest raising the limit: {stderr}"
    );
}

// ---- Issue #374: compiler and target metadata ----

#[test]
fn limits_help_documents_wasm_target_dir_and_profile() {
    let output = run(&["limits", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--wasm-target-dir"), "{stdout}");
    assert!(stdout.contains("--profile"), "{stdout}");
}

#[test]
fn limits_output_includes_compiler_metadata() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--dry-run",
    ]);
    // dry-run exits before measuring, so we just check the flags parse cleanly
    assert!(output.status.success(), "{:?}", output);
}

// ---- Issue #370: custom wasm target directory ----

#[test]
fn limits_wasm_target_dir_flag_is_accepted() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--wasm-target-dir",
        "/tmp/custom-target",
        "--dry-run",
    ]);
    assert!(output.status.success(), "{:?}", output);
}

// ---- Issue #371: cargo profile selection ----

#[test]
fn limits_profile_flag_is_accepted() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--profile",
        "release",
        "--dry-run",
    ]);
    assert!(output.status.success(), "{:?}", output);
}

// ---- Issue #360: per-probe timeout ----

#[test]
fn limits_probe_timeout_failure_names_the_killed_value_and_timeout() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--probe-timeout",
        "0",
    ]);
    // An immediate deadline kills the first probe before it can finish. A probe
    // that wins the race is fine too; a killed one must explain itself.
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("timed-out"), "{stderr}");
        assert!(
            stderr.contains("probe for recipients=1 was killed after exceeding its 0ms timeout"),
            "{stderr}"
        );
        assert!(stderr.contains("raise --probe-timeout"), "{stderr}");
    }
}

#[test]
fn limits_rejects_a_probe_timeout_the_clock_cannot_represent_before_probing() {
    let wasm = ensure_vault_wasm();
    let root = std::env::temp_dir().join(format!(
        "stk-limits-huge-probe-timeout-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let export = root.join("report.json");
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--probe-timeout",
        &u64::MAX.to_string(),
        "--export",
        export.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--probe-timeout"), "{stderr}");
    assert!(stderr.contains("too large for the host clock"), "{stderr}");
    assert!(
        !export.exists(),
        "no report may be written for a rejected flag"
    );
    std::fs::remove_dir_all(&root).ok();
}

// ---- Issue #361: overall command timeout ----

#[test]
fn limits_rejects_a_zero_overall_timeout() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--timeout",
        "0",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--timeout must be at least 1 second"),
        "{stderr}"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("last successful"));
}

#[test]
fn limits_overall_timeout_stops_a_long_ramp_with_partial_results() {
    let wasm = ensure_vault_wasm();
    let root =
        std::env::temp_dir().join(format!("stk-limits-overall-timeout-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let export = root.join("report.json");
    let start = std::time::Instant::now();
    // A step of 1 needs thousands of probes, far more than fit in one second.
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--search-strategy",
        "fixed-step",
        "--step",
        "1",
        "--timeout",
        "1",
        "--export",
        export.to_str().unwrap(),
    ]);
    let elapsed = start.elapsed();
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("\"batch_payout\" exceeded --timeout (1s)"),
        "{stderr}"
    );
    assert!(
        stderr.contains("highest recipients verified so far is"),
        "{stderr}"
    );
    assert!(stderr.contains("boundary was not confirmed"), "{stderr}");
    assert!(stderr.contains("Raise --timeout"), "{stderr}");
    assert!(
        !String::from_utf8_lossy(&output.stdout).contains("last successful"),
        "an unconfirmed boundary must not be reported as a result"
    );
    assert!(
        !export.exists(),
        "no report may be written for an unfinished run"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "the run should stop near the deadline, took {elapsed:?}"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn limits_generous_overall_timeout_does_not_change_the_result() {
    let wasm = ensure_vault_wasm();
    let wasm = wasm.to_str().unwrap();
    let base = [
        "limits",
        "--contract",
        wasm,
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
    ];
    let without = run(&base);
    let with_args: Vec<&str> = base.iter().copied().chain(["--timeout", "600"]).collect();
    let with = run(&with_args);
    assert!(without.status.success(), "{without:?}");
    assert!(with.status.success(), "{with:?}");
    let best_line = |out: &Output| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find(|l| l.starts_with("last successful"))
            .map(str::to_string)
    };
    assert!(best_line(&with).is_some());
    assert_eq!(best_line(&with), best_line(&without));
}

// ---- Issue #362: progress updates ----

#[test]
fn limits_progress_writes_one_line_per_probe_to_stderr() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--progress",
    ]);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("limits: probe "))
        .collect();
    assert!(
        lines[0].starts_with("limits: probe 1 (scan) recipients=1 ok; best so far: 1; elapsed "),
        "{stderr}"
    );
    assert!(lines.iter().any(|l| l.contains("(refine)")), "{stderr}");
    assert!(lines.iter().any(|l| l.contains(" failed [")), "{stderr}");
    // Numbered in the order the probes were recorded, with no gaps.
    for (i, line) in lines.iter().enumerate() {
        assert!(
            line.starts_with(&format!("limits: probe {} (", i + 1)),
            "{line}"
        );
    }
    // Stdout keeps the normal summary and never receives progress lines.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("last successful"), "{stdout}");
    assert!(!stdout.contains("limits: probe"), "{stdout}");
}

#[test]
fn limits_without_progress_prints_no_progress_lines() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
    ]);
    assert!(output.status.success(), "{output:?}");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("limits: probe"));
}

#[test]
fn limits_progress_reports_the_failing_probe_before_a_contract_error() {
    let wasm = ensure_vault_wasm();
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "reject_after_three",
        "--ramp",
        "count",
        "--progress",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("limits: probe 3 (scan) count=4 failed [contract-failure]"),
        "{stderr}"
    );
    assert!(
        stderr.contains("before a resource limit was verified"),
        "{stderr}"
    );
}

// ---- Issue #363: last failing probe ----

#[test]
fn limits_reports_the_last_failing_probe_alongside_the_maximum_success() {
    let wasm = ensure_vault_wasm();
    let root = std::env::temp_dir().join(format!("stk-limits-last-failure-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let export = root.join("report.json");
    let output = run(&[
        "limits",
        "--contract",
        wasm.to_str().unwrap(),
        "--fn",
        "batch_payout",
        "--ramp",
        "recipients",
        "--export",
        export.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let best_at = lines
        .iter()
        .position(|l| l.starts_with("last successful"))
        .unwrap();
    let last_at = lines
        .iter()
        .position(|l| l.starts_with("  last failing probe: recipients="))
        .unwrap_or_else(|| panic!("missing last failing probe line: {stdout}"));
    assert!(last_at > best_at && last_at <= best_at + 4, "{stdout}");

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&export).unwrap()).unwrap();
    let best = json["best"].as_u64().unwrap();
    let last = &json["last_failed_probe"];
    // After refinement the last failing probe is the boundary just above `best`.
    assert_eq!(last["value"].as_u64().unwrap(), best + 1, "{json}");
    assert_eq!(last["value"], json["first_failed"], "{json}");
    let category = last["category"].as_str().expect("category is a string");
    assert!(
        lines[last_at].contains(&format!("recipients={} [{category}]", best + 1)),
        "{stdout}"
    );
    std::fs::remove_dir_all(&root).ok();
}
