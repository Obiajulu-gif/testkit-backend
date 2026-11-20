//! Resource-ceiling discovery for compiled Soroban contracts.
//!
//! This module owns the `limits` command and the hidden `__limits-probe`
//! child entry point it spawns. The two halves cooperate through process
//! isolation, whose full lifecycle is documented on [`run`]; each probe's
//! result is classified with [`ProbeFailureCategory`] and every report â
//! human-readable or machine-readable â is rendered from a single
//! [`LimitsSummary`] so the formats cannot disagree.

use std::collections::BTreeMap;
use std::panic;
use std::path::Path;
use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant};

use clap::Args;
use soroban_sdk::testutils::{Address as _, MockAuth, MockAuthInvoke};
use soroban_sdk::token::StellarAssetClient;
use soroban_sdk::xdr::{ScSpecEntry, ScSpecFunctionInputV0, ScSpecFunctionV0, ScSpecTypeDef};
use soroban_sdk::{Address, Env, IntoVal, Symbol, TryFromVal, Val, Vec as SVec};

use super::CliError;

/// Measurement limitations warning reported across stdout and machine-readable outputs.
pub const MEASUREMENT_LIMITATIONS: &str =
    "ledger reads/writes and transaction size are not measured by this command; see --help";

/// Soroban per-transaction CPU instruction limit on mainnet.
pub const SOROBAN_CPU_INSN_LIMIT: u64 = 100_000_000;

/// Soroban per-transaction memory limit in bytes on mainnet (40 MiB).
pub const SOROBAN_MEMORY_LIMIT: u64 = 41_943_040;

/// Help text explaining how to interpret local instruction measurements.
pub const INSTRUCTION_INTERPRETATION: &str =
    "instruction counts are Soroban host CPU instructions, not wall-clock time;      one unit is roughly one basic WASM interpreter operation;      the per-transaction ceiling on Stellar mainnet is 100,000,000 instructions;      local measurements omit transaction-envelope overhead and may read      slightly lower than a network simulation of the same call";

/// Built-in upper bound for the doubling search. If the contract still
/// succeeds here the run stops and reports that the ceiling may be higher,
/// rather than ramping forever.
pub const SEARCH_UPPER_BOUND: u32 = 1 << 24;

/// Largest accepted `--jobs`. Each concurrent probe is a full child process,
/// so the bound keeps memory and CPU use predictable.
const MAX_PROBE_JOBS: usize = 8;

/// The standardized reason a single probe did not succeed.
///
/// Every failure category is reported verbatim in the human-readable summary,
/// the refinement audit, and the JSON export, so a failure can be triaged the
/// same way regardless of which output format a CI job happens to consume.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum ProbeFailureCategory {
    /// The child process ran to completion and exited non-zero: the contract
    /// either rejected the ramp value with its own validation error or
    /// exhausted a host resource budget.
    ExitFailure,
    /// A contract rejected or trapped before exhausting a resource budget.
    ContractFailure,
    /// The probe exceeded `--probe-timeout` and was killed by the parent
    /// before it could finish.
    TimedOut,
    /// The child process was terminated by a signal â the host abort that
    /// real WASM execution can cause rather than a normal exit. On such a
    /// run the failure is real, but the ramp value reported may reflect the
    /// abort rather than a clean resource boundary.
    Abort,
    /// The isolated probe process could not be spawned at all; this is a
    /// host/tooling problem, not a statement about the contract.
    SpawnFailure,
}

impl ProbeFailureCategory {
    /// The stable, machine-readable category name (also the JSON key).
    pub fn label(self) -> &'static str {
        match self {
            ProbeFailureCategory::ExitFailure => "exit-failure",
            ProbeFailureCategory::ContractFailure => "contract-failure",
            ProbeFailureCategory::TimedOut => "timed-out",
            ProbeFailureCategory::Abort => "abort",
            ProbeFailureCategory::SpawnFailure => "spawn-failure",
        }
    }

    /// A one-line, actionable explanation of what the category implies for
    /// the discovered ceiling.
    pub fn guidance(self) -> &'static str {
        match self {
            ProbeFailureCategory::ExitFailure => {
                "the probe exited non-zero; check its stderr to distinguish rejection from exhaustion"
            }
            ProbeFailureCategory::ContractFailure => {
                "the contract rejected or trapped at this value before any verified resource ceiling; inspect the contract logic"
            }
            ProbeFailureCategory::TimedOut => {
                "raise --probe-timeout to confirm whether the ceiling reflects a hang or a real resource limit"
            }
            ProbeFailureCategory::Abort => {
                "the host aborted during WASM execution; re-run to confirm the boundary is stable"
            }
            ProbeFailureCategory::SpawnFailure => {
                "relocate/reinstall the soroban-testkit binary because probes could not be spawned"
            }
        }
    }
}

impl std::fmt::Display for ProbeFailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
/// Which host resource was the binding constraint that stopped the ramp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResourceBoundary {
    /// CPU instruction count was closer to its mainnet limit.
    Instructions,
    /// Memory bytes was closer to its mainnet limit.
    Memory,
    /// Neither resource approached its limit; the ramp was stopped by contract
    /// logic, the built-in search ceiling, or an indeterminate cause.
    Other,
}

impl ResourceBoundary {
    /// A one-line explanation of what this boundary means for the reported ceiling.
    pub fn explanation(self) -> &'static str {
        match self {
            ResourceBoundary::Instructions => {
                "CPU instructions were the binding resource; the contract is compute-bound"
            }
            ResourceBoundary::Memory => {
                "memory bytes were the binding resource; the contract is memory-bound"
            }
            ResourceBoundary::Other => {
                "neither resource approached its limit; the ceiling reflects contract logic,                  the built-in search upper bound, or an indeterminate cause"
            }
        }
    }
}

impl std::fmt::Display for ResourceBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceBoundary::Instructions => write!(f, "instructions"),
            ResourceBoundary::Memory => write!(f, "memory"),
            ResourceBoundary::Other => write!(f, "other"),
        }
    }
}

/// A named configuration for a common Soroban batch workload.
///
/// Pass via `--preset`. Available presets:
/// - `batch-payout`   -- `--fn batch_payout --ramp recipients`
/// - `token-transfer` -- `--fn transfer --ramp amount`
/// - `multi-mint`     -- `--fn mint --ramp count`
///
/// Explicit `--fn`/`--ramp`/`--arg` flags always override the preset's defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RampPreset {
    /// Targets `batch_payout(recipients, ...)`.
    BatchPayout,
    /// Targets `transfer(amount, ...)`.
    TokenTransfer,
    /// Targets `mint(count, ...)`.
    MultiMint,
}

impl RampPreset {
    /// Parse a preset name, returning an error listing valid names on failure.
    pub fn from_name(name: &str) -> Result<Self, CliError> {
        match name {
            "batch-payout" => Ok(RampPreset::BatchPayout),
            "token-transfer" => Ok(RampPreset::TokenTransfer),
            "multi-mint" => Ok(RampPreset::MultiMint),
            other => Err(CliError(format!(
                "unknown preset {other:?}; available presets: batch-payout, token-transfer, multi-mint"
            ))),
        }
    }

    /// The contract function name this preset targets.
    pub fn function(&self) -> &'static str {
        match self {
            RampPreset::BatchPayout => "batch_payout",
            RampPreset::TokenTransfer => "transfer",
            RampPreset::MultiMint => "mint",
        }
    }

    /// The ramp parameter name this preset uses.
    pub fn ramp(&self) -> &'static str {
        match self {
            RampPreset::BatchPayout => "recipients",
            RampPreset::TokenTransfer => "amount",
            RampPreset::MultiMint => "count",
        }
    }

    /// Default `NAME=VALUE` arg overrides this preset recommends.
    pub fn default_args(&self) -> &'static [&'static str] {
        match self {
            RampPreset::BatchPayout | RampPreset::TokenTransfer | RampPreset::MultiMint => &[],
        }
    }
}

/// A checked-in resource ceiling for a `limits` run.
///
/// Unlike [`Baseline`] (which records a previous run's actual results),
/// a `ResourceBudget` is a human-authored file specifying the *maximum*
/// CPU instructions, memory bytes, or *minimum* ramp ceiling the contract must
/// satisfy. Check one into the repository alongside the contract to guard
/// against silent resource regressions in CI.
///
/// Budget files use a `key=value` text format with optional `#` comment lines.
/// Every field is optional; omitted fields are not checked.
///
/// # Example budget file
///
/// ```text
/// # maximum allowed CPU instructions at the ramp ceiling
/// max_instructions=60000000
/// # maximum allowed memory bytes
/// max_memory_bytes=20000000
/// # minimum ramp ceiling the contract must sustain
/// min_best=100
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Maximum CPU instructions allowed at the discovered ramp ceiling.
    pub max_instructions: Option<u64>,
    /// Maximum memory bytes allowed at the discovered ramp ceiling.
    pub max_memory_bytes: Option<u64>,
    /// Minimum ramp ceiling the contract must achieve.
    pub min_best: Option<u32>,
}

impl ResourceBudget {
    /// Parse a resource budget from its `key=value` file contents.
    pub fn parse(contents: &str) -> Result<Self, CliError> {
        let mut max_instructions = None;
        let mut max_memory_bytes = None;
        let mut min_best = None;

        for (line_no, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(CliError(format!(
                    "malformed budget file at line {}: expected key=value or # comment, got {line:?}",
                    line_no + 1
                )));
            };
            match key.trim() {
                "max_instructions" => {
                    max_instructions = Some(value.trim().parse::<u64>().map_err(|err| {
                        CliError(format!(
                            "malformed budget `max_instructions` value {:?}: {err}",
                            value
                        ))
                    })?)
                }
                "max_memory_bytes" => {
                    max_memory_bytes = Some(value.trim().parse::<u64>().map_err(|err| {
                        CliError(format!(
                            "malformed budget `max_memory_bytes` value {:?}: {err}",
                            value
                        ))
                    })?)
                }
                "min_best" => {
                    min_best = Some(value.trim().parse::<u32>().map_err(|err| {
                        CliError(format!(
                            "malformed budget `min_best` value {:?}: {err}",
                            value
                        ))
                    })?)
                }
                other => {
                    return Err(CliError(format!(
                        "malformed budget file at line {}: unknown key {other:?};                          valid keys are max_instructions, max_memory_bytes, min_best",
                        line_no + 1
                    )))
                }
            }
        }

        Ok(ResourceBudget {
            max_instructions,
            max_memory_bytes,
            min_best,
        })
    }
}

/// The outcome of a single isolated probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// The probe exited zero: the ramp value is supported.
    Succeeded,
    /// The probe failed, classified for the report.
    Failed(ProbeFailureCategory),
}

impl ProbeOutcome {
    /// Whether the probe succeeded, for the binary search's yes/no decision.
    pub fn succeeded(self) -> bool {
        matches!(self, ProbeOutcome::Succeeded)
    }
}

/// Probe child errors have a stable marker only when the SDK reported a
/// non-budget invocation error. Generic non-zero exits remain ambiguous.
fn classify_probe_exit_stderr(stderr: &str) -> ProbeFailureCategory {
    if stderr
        .lines()
        .any(|line| line.starts_with("error: contract-failure:"))
    {
        ProbeFailureCategory::ContractFailure
    } else {
        ProbeFailureCategory::ExitFailure
    }
}

/// `serde(skip_serializing_if)` helper: omitted for an empty failure map.
/// The first non-empty line of a probe's stderr, so one-line summaries stay
/// one line even when a probe prints a multi-line panic message.
fn first_line(text: &str) -> &str {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
}

fn map_is_empty<K, V>(map: &BTreeMap<K, V>) -> bool {
    map.is_empty()
}

/// `serde(skip_serializing_if)` helper: omitted for an empty vec.
fn vec_is_empty<T>(v: &[T]) -> bool {
    v.is_empty()
}

/// The most recent probe that failed during a run (issue #363).
///
/// [`LimitsSummary::first_failed`] is the refined boundary, and
/// [`LimitsSummary::first_failed_probe_stderr`] comes from the *first* probe
/// that failed chronologically -- usually a doubling probe far above the
/// boundary. This record is the failing probe that ran last, which after
/// refinement is the one sitting right above the maximum success, so its
/// category and stderr explain why the reported ceiling is where it is.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FailedProbe {
    /// The ramp value the probe was run with.
    pub value: u32,
    /// Why it failed.
    pub category: ProbeFailureCategory,
    /// Stderr captured from that probe (or the timeout note for a killed
    /// probe), when there was any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
}

/// One recorded probe sample, included in the `--export-samples` output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProbeSample {
    pub value: u32,
    pub succeeded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_category: Option<ProbeFailureCategory>,
    /// Wall-clock duration of this probe in milliseconds, measured from spawn
    /// to result classification. Present for every probe when `--record-probe-duration`
    /// is set; `None` when the flag is omitted (the default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// The single source of truth for one `limits` run.
///
/// The terminal summary ([`LimitsSummary::render_text`]) and the JSON export
/// ([`export_results`]) are both rendered from this value, so the two formats
/// can never disagree about the discovered boundary, the failing value, or
/// why probes failed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LimitsSummary {
    /// Sanitized contract path (see [`sanitize_report_path`]).
    pub contract: String,
    pub function: Option<String>,
    pub ramp: Option<String>,
    /// The highest ramp value that succeeded.
    pub best: u32,
    /// CPU instructions consumed at `best`.
    pub instructions: u64,
    /// Memory bytes consumed at `best`.
    pub memory_bytes: u64,
    /// The first ramp value that failed, if one was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_failed: Option<u32>,
    /// Why that first failure happened, standardized for every report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_failure_category: Option<ProbeFailureCategory>,
    /// Stderr captured from the first failing probe, providing
    /// actionable context for why the ceiling was hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_failed_probe_stderr: Option<String>,
    /// The failing probe that ran last, reported alongside `best` (see
    /// [`FailedProbe`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_failed_probe: Option<FailedProbe>,
    /// How many probes failed in each category over the whole run.
    #[serde(default, skip_serializing_if = "map_is_empty")]
    pub probe_failure_counts: BTreeMap<ProbeFailureCategory, usize>,
    /// Whether the search stopped at [`SEARCH_UPPER_BOUND`] without finding a
    /// failure.
    pub hit_upper_bound: bool,
    /// Per-probe timeout the run used, for the timeout guidance line.
    pub probe_timeout_secs: u64,
    /// Boundary-refinement audit trail, when a failure was bracketed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit: Option<RefinementAudit>,
    pub measurement_limitations: String,
    /// Actionable instructions for reproducing the reported maximum locally.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproduction: Option<String>,
    /// Which host resource was the binding constraint that stopped the ramp.
    /// Computed from the fraction of Soroban mainnet limits consumed at `best`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped_by: Option<ResourceBoundary>,
    /// All individual probe samples recorded during the run, in call order.
    /// Populated only when `--export-samples` is requested.
    #[serde(default, skip_serializing_if = "vec_is_empty")]
    pub probe_samples: Vec<ProbeSample>,
    /// Additional parameters ramped in lockstep with the primary ramp.
    #[serde(default, skip_serializing_if = "vec_is_empty")]
    pub co_ramp: Vec<String>,
    /// Compiler and WASM-target metadata captured at run time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compiler_meta: Option<CompilerMeta>,
    /// Custom wasm target directory, when `--wasm-target-dir` was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wasm_target_dir: Option<String>,
    /// Cargo profile used for the build, when `--profile` was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
}

impl LimitsSummary {
    /// Renders the human-readable summary. This is the only place the text
    /// output is produced, keeping it in lockstep with the JSON export.
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        let ramp_str = self.ramp.as_deref().unwrap_or("");
        let function_str = self.function.as_deref().unwrap_or("");
        let ramp_display = if self.co_ramp.is_empty() {
            ramp_str.to_string()
        } else {
            format!("{} (+ {})", ramp_str, self.co_ramp.join(", "))
        };
        out.push_str(&format!(
            "last successful {} for {:?}: {}\n",
            ramp_display, function_str, self.best
        ));
        out.push_str(&format!("  instructions: {}\n", self.instructions));
        out.push_str(&format!("  memory bytes: {}\n", self.memory_bytes));
        if let Some(failed) = self.first_failed {
            out.push_str(&format!("  first failed {} value: {failed}", ramp_str));
            if let Some(category) = self.first_failure_category {
                out.push_str(&format!(" [{category}]"));
            }
            out.push('\n');
        }
        if let Some(last) = &self.last_failed_probe {
            out.push_str(&format!(
                "  last failing probe: {}={} [{}]",
                ramp_str, last.value, last.category
            ));
            if let Some(stderr) = last
                .stderr
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                out.push_str(&format!(": {}", first_line(stderr)));
            }
            out.push('\n');
        }
        if !self.probe_failure_counts.is_empty() {
            let counts = self
                .probe_failure_counts
                .iter()
                .map(|(category, count)| format!("{category}: {count}"))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("  probe failures: {counts}\n"));
        }
        if let Some(stderr) = &self.first_failed_probe_stderr {
            let trimmed = stderr.trim();
            if !trimmed.is_empty() {
                out.push_str(&format!("  first failing probe stderr: {trimmed}\n"));
            }
        }
        if let Some(audit) = &self.audit {
            out.push_str(&format!(
                "  boundary refinement: probed {} values between {} and {} (final boundary: [{}, {}])\n",
                audit.steps.len(),
                audit.initial_low,
                audit.initial_high,
                audit.final_success,
                audit.final_failure,
            ));
        }
        out.push_str(&format!("  ({})\n", self.measurement_limitations));
        out.push_str(&format!("  ({})\n", INSTRUCTION_INTERPRETATION));
        if let Some(boundary) = self.stopped_by {
            out.push_str(&format!(
                "  resource boundary: {} -- {}\n",
                boundary,
                boundary.explanation()
            ));
        }
        if let Some(repro) = &self.reproduction {
            out.push_str(&format!("  reproduce locally: {repro}\n"));
        }
        if self
            .probe_failure_counts
            .contains_key(&ProbeFailureCategory::Abort)
        {
            out.push_str(
                "  note: a host abort may reflect a contract trap rather than a resource limit; \
                 inspect the function and re-run to confirm\n",
            );
        }
        if let Some(meta) = &self.compiler_meta {
            out.push_str(&format!(
                "  compiler: {} (target: {})\n",
                meta.rustc_version, meta.wasm_target
            ));
        }
        if let Some(dir) = &self.wasm_target_dir {
            out.push_str(&format!("  wasm target dir: {dir}\n"));
        }
        if let Some(profile) = &self.profile {
            out.push_str(&format!("  cargo profile: {profile}\n"));
        }
        if self.hit_upper_bound {
            out.push_str(&format!(
                "  note: search stopped at the built-in upper bound ({SEARCH_UPPER_BOUND}); the \
                 contract may support higher ramp values\n"
            ));
        }
        if self
            .probe_failure_counts
            .contains_key(&ProbeFailureCategory::TimedOut)
        {
            out.push_str(&format!(
                "  note: at least one probe was killed for exceeding --probe-timeout ({}s); the \
                 discovered ceiling may reflect a hang rather than a real resource limit â rerun \
                 with a larger --probe-timeout to check\n",
                self.probe_timeout_secs
            ));
        }
        out
    }
}

/// A single step in a numeric boundary refinement search.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RefinementStep {
    pub step: usize,
    pub probe_value: u32,
    pub succeeded: bool,
    /// Why the probe failed, when it did. Uses the standardized
    /// [`ProbeFailureCategory`] vocabulary so every report agrees.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_category: Option<ProbeFailureCategory>,
    pub window_low: u32,
    pub window_high: u32,
}

/// Structured audit trail documenting boundary refinement progress and final window.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RefinementAudit {
    pub initial_low: u32,
    pub initial_high: u32,
    pub steps: Vec<RefinementStep>,
    pub final_success: u32,
    pub final_failure: u32,
}

/// Compiler and target metadata captured at the start of a `limits` run.
///
/// Recorded in [`LimitsSummary`] so results files carry the toolchain context
/// they were produced with, making regression comparisons across compiler
/// upgrades auditable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct CompilerMeta {
    /// Output of `rustc --version` at run time.
    pub rustc_version: String,
    /// The WASM target triple used for Soroban contracts.
    pub wasm_target: String,
}

/// Captures compiler and WASM-target metadata for the current toolchain.
fn collect_compiler_meta() -> CompilerMeta {
    let rustc_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    CompilerMeta {
        rustc_version,
        wasm_target: "wasm32v1-none".to_string(),
    }
}

/// A reusable comparison between a `limits` run and a saved baseline.
///
/// Extracted from the private `compare_to_baseline` so callers that need
/// programmatic access to regression details (CI scripts, test helpers) do
/// not have to parse human-readable output.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineComparison {
    pub ramp: String,
    pub function: String,
    pub baseline_best: u32,
    pub current_best: u32,
    pub baseline_instructions: u64,
    pub current_instructions: u64,
    pub baseline_memory_bytes: u64,
    pub current_memory_bytes: u64,
    pub tolerance_pct: f64,
    regressions: Vec<String>,
}

impl BaselineComparison {
    /// Compares `current_*` metrics against `baseline`, applying
    /// `tolerance_pct` to instruction/memory growth.
    ///
    /// Returns an error only when the `function`/`ramp` combination does not
    /// match the baseline (comparing across different call sites is not
    /// meaningful).
    pub fn new(
        baseline: &Baseline,
        ramp: &str,
        function: &str,
        best: u32,
        instructions: u64,
        memory_bytes: u64,
        tolerance_pct: f64,
    ) -> Result<Self, CliError> {
        if baseline.function != function || baseline.ramp != ramp {
            return Err(CliError(format!(
                "baseline was recorded for {:?}/{:?}, not {function:?}/{ramp:?}; comparing \
                 across different functions or ramp parameters isn't meaningful",
                baseline.function, baseline.ramp
            )));
        }
        let mut regressions = Vec::new();
        if best < baseline.best {
            regressions.push(format!(
                "{ramp} ceiling dropped from {} to {best}",
                baseline.best
            ));
        }
        Self::check_metric(
            "instructions",
            baseline.instructions,
            instructions,
            tolerance_pct,
            &mut regressions,
        );
        Self::check_metric(
            "memory bytes",
            baseline.memory_bytes,
            memory_bytes,
            tolerance_pct,
            &mut regressions,
        );
        Ok(BaselineComparison {
            ramp: ramp.to_string(),
            function: function.to_string(),
            baseline_best: baseline.best,
            current_best: best,
            baseline_instructions: baseline.instructions,
            current_instructions: instructions,
            baseline_memory_bytes: baseline.memory_bytes,
            current_memory_bytes: memory_bytes,
            tolerance_pct,
            regressions,
        })
    }

    /// Returns `true` if any metric regressed beyond tolerance.
    pub fn has_regression(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// Human-readable regression descriptions, one per violated metric.
    pub fn regressions(&self) -> &[String] {
        &self.regressions
    }

    fn check_metric(name: &str, old: u64, new: u64, tolerance_pct: f64, out: &mut Vec<String>) {
        if new <= old {
            return;
        }
        let growth_pct = ((new - old) as f64 / old.max(1) as f64) * 100.0;
        if growth_pct <= tolerance_pct {
            return;
        }
        out.push(format!(
            "{name} grew from {old} to {new} ({growth_pct:.1}%, over the {tolerance_pct:.1}% tolerance)"
        ));
    }
}

/// Which strategy to use during the initial scan phase of a `limits` run.
///
/// Pass via `--search-strategy`. The binary-refinement phase that narrows the
/// bracket found by the scan always runs regardless of strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// Double the ramp value on each step: 1, 2, 4, 8, … (default). Finds a
    /// failing bracket in O(log N) probes; best when the ceiling is unknown.
    #[default]
    Doubling,
    /// Advance by a fixed step on each probe. The step size is set by
    /// `--step` (default 1). Use when you need a fully predictable probe
    /// sequence or want to explore a narrow range exhaustively.
    FixedStep,
}

impl std::str::FromStr for SearchStrategy {
    type Err = CliError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "doubling" => Ok(SearchStrategy::Doubling),
            "fixed-step" => Ok(SearchStrategy::FixedStep),
            other => Err(CliError(format!(
                "unknown search strategy {other:?}; available strategies: doubling, fixed-step"
            ))),
        }
    }
}

/// Arguments for `soroban-testkit limits`.
#[derive(Args)]
pub struct LimitsArgs {
    /// Path to an existing compiled contract .wasm file. Build successfully before
    /// running limits; this command does not rebuild contracts.
    #[arg(long, value_name = "PATH")]
    contract: std::path::PathBuf,
    /// The contract function to ramp. Required unless --preset is given.
    #[arg(long = "fn", value_name = "NAME", required_unless_present = "preset")]
    function: Option<String>,
    /// The parameter to increase on each attempt. Must exactly match an input
    /// name in the selected function's contract ABI; checked before probing.
    /// Required unless --preset is given.
    #[arg(long, value_name = "PARAM", required_unless_present = "preset")]
    ramp: Option<String>,
    /// Additional parameters to ramp in lockstep with --ramp. Each listed
    /// parameter is set to the same value as the primary ramp parameter on
    /// every probe. Repeatable. Must each match an input name in the ABI.
    #[arg(long = "co-ramp", value_name = "PARAM")]
    co_ramp: Vec<String>,
    /// Explicit value for a non-ramped parameter, as NAME=VALUE. Repeatable.
    /// Scalars use their Soroban type; Bytes uses 0x-prefixed hexadecimal.
    #[arg(long = "arg", value_name = "NAME=VALUE")]
    arg_values: Vec<String>,
    /// Kill and treat as a failure any single isolated probe that runs longer
    /// than this many seconds. Guards against a ramp value that hangs the
    /// host (rather than erroring or aborting) turning `limits` into an
    /// infinite wait. A killed probe is classified `timed-out`, and its report
    /// names the ramp value and the timeout that was exceeded.
    #[arg(long, value_name = "SECONDS", default_value_t = 30)]
    probe_timeout: u64,
    /// Fail the whole command if probing takes longer than this many seconds
    /// in total (at least 1; off by default). Each probe waits at most for the
    /// time left, so the run stops close to the deadline even with a large
    /// --probe-timeout. A probe cut short by this deadline is not counted as a
    /// resource ceiling; the error reports the best value verified so far.
    #[arg(long, value_name = "SECONDS")]
    timeout: Option<u64>,
    /// Print one progress line per probe to stderr while ramping: the phase
    /// (scan or refine), probe count, ramp value, outcome, elapsed time and the
    /// best value so far. Stdout and the --export JSON are unchanged.
    #[arg(long, default_value_t = false)]
    progress: bool,
    /// Probe up to this many ramp values at once while doubling to find the
    /// first failing value (1 to 8, default 1 = sequential). Results are read
    /// back in ramp order, so the reported boundary does not depend on which
    /// probe finishes first. Binary refinement between the bounds always runs
    /// one probe at a time. Probes past the first failure still run to completion
    /// but are discarded, so extra work is bounded by `--jobs - 1` probes.
    #[arg(long, value_name = "N", default_value_t = 1)]
    jobs: usize,
    /// Compare this run's result against a baseline previously written by
    /// `--save-baseline`. Exits non-zero if the discovered ceiling
    /// regressed (a lower ramp value, or higher instructions/memory) by
    /// more than `--baseline-tolerance-pct`.
    #[arg(long, value_name = "PATH")]
    baseline: Option<std::path::PathBuf>,
    /// How much instructions/memory are allowed to grow (as a percentage
    /// of the baseline value) before `--baseline` reports a regression.
    /// The ramp ceiling itself allows no tolerance: any decrease is a
    /// regression, since it means the contract now handles *fewer*
    /// recipients (or whatever the ramp parameter represents) than before.
    #[arg(long, value_name = "PCT", default_value_t = 5.0)]
    baseline_tolerance_pct: f64,
    /// Write this run's result to PATH as a new baseline for future
    /// `--baseline` comparisons. Combinable with `--baseline` itself, to
    /// compare against the old baseline and then update it in one run.
    #[arg(long, value_name = "PATH")]
    save_baseline: Option<std::path::PathBuf>,
    /// Export this run's result to PATH as a JSON file for CI artifacts.
    /// Contains the discovered ceiling, instructions, and memory usage.
    #[arg(long, value_name = "PATH")]
    export: Option<std::path::PathBuf>,
    /// Apply a named preset for a common batch workload, supplying defaults for
    /// --fn and --ramp. Explicit --fn/--ramp/--arg flags always override the
    /// preset's defaults. Available presets: batch-payout, token-transfer, multi-mint.
    #[arg(long, value_name = "PRESET")]
    preset: Option<String>,
    /// Compare this run's result against a checked-in resource budget file.
    /// The budget file specifies max_instructions, max_memory_bytes, and/or
    /// min_best as key=value lines (with optional # comments). Exits non-zero
    /// if any limit is exceeded.
    #[arg(long, value_name = "PATH")]
    budget: Option<std::path::PathBuf>,
    /// Export every individual probe sample to PATH as a JSON array. Each
    /// entry records the ramp value, whether it succeeded, and the failure
    /// category when it failed. Useful for plotting the full resource curve.
    #[arg(long, value_name = "PATH")]
    export_samples: Option<std::path::PathBuf>,
    /// Print the planned probe sequence without executing any probes and
    /// exit. Shows the ramp values that would be attempted in the doubling
    /// phase, up to the first theoretical failure bracket.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    /// Abort with an error if the total number of probe invocations exceeds
    /// this limit. Protects against unexpectedly large search spaces in CI.
    /// Set to 0 to disable the limit (the default).
    #[arg(long, value_name = "N", default_value_t = 0)]
    max_probes: usize,
    /// Override the directory where Cargo places the compiled .wasm artifact.
    /// Useful when the contract was built with a custom `--target-dir` or when
    /// running limits against a pre-built artifact in a non-standard location.
    #[arg(long, value_name = "PATH")]
    wasm_target_dir: Option<std::path::PathBuf>,
    /// Cargo profile used when building the contract (e.g. `release`, `dev`).
    /// Recorded in the output so results files carry the build context they
    /// were produced with. Does not trigger a rebuild; the .wasm at `--contract`
    /// is used as-is.
    #[arg(long, value_name = "PROFILE")]
    profile: Option<String>,
    /// Strategy used during the initial scan phase that brackets the first
    /// failing ramp value. `doubling` (the default) doubles on each probe:
    /// 1, 2, 4, 8, … `fixed-step` advances by a fixed amount set with `--step`.
    #[arg(long, value_name = "STRATEGY", default_value = "doubling")]
    search_strategy: SearchStrategy,
    /// Step size for `--search-strategy fixed-step`. The probe sequence will be
    /// `--step`, `2 × --step`, `3 × --step`, … up to the search upper bound.
    /// Ignored unless `--search-strategy fixed-step` is also set.
    #[arg(long, value_name = "N", default_value_t = 1)]
    step: u32,
    /// Record the wall-clock duration of every probe in the samples output.
    /// When set, each `ProbeSample` in `--export-samples` gains a `duration_ms`
    /// field with the elapsed milliseconds from spawn to result classification.
    #[arg(long, default_value_t = false)]
    record_probe_duration: bool,
}

/// Hidden: runs exactly one probe (a single ramp value) and reports its
/// outcome via exit code. See [`run`]'s doc comment for why this exists
/// as a separate process rather than an in-process call.
#[derive(Args)]
pub struct ProbeArgs {
    #[arg(long)]
    contract: std::path::PathBuf,
    #[arg(long = "fn")]
    function: String,
    #[arg(long)]
    ramp: String,
    #[arg(long)]
    value: u32,
    #[arg(long = "arg", value_name = "NAME=VALUE")]
    arg_values: Vec<String>,
    #[arg(long = "co-ramp")]
    co_ramp: Vec<String>,
    #[arg(long)]
    wasm_target_dir: Option<std::path::PathBuf>,
    #[arg(long)]
    profile: Option<String>,
}

/// Empirically discovers a resource ceiling by invoking `--fn` with an
/// increasing `--ramp` parameter â natively, with no network access â
/// until it fails, then reports the last successful value plus the CPU
/// instructions and memory consumed at that point.
///
/// # Subprocess isolation lifecycle
///
/// Ramping deliberately drives a real, compiled `.wasm` contract past its
/// resource limits. Verified empirically: when that failure happens
/// *during actual WASM execution* (as opposed to a native, non-WASM
/// `#[contract]` registration), the host can abort the whole process
/// (`thread caused non-unwinding panic. aborting.`) rather than return an
/// error or unwind â neither `catch_unwind` nor `try_invoke_contract`
/// (the SDK's own no-panic call path) prevents this for the WASM target.
/// A probe is therefore never invoked in-process. The full lifecycle is:
///
/// 1. **Validate first, in-process.** [`load`] and [`build_args`] check the
///    function spec, the ramp parameter, and every other parameter's type
///    before any child is spawned. This is type/spec checking, not WASM
///    execution, so it carries none of the abort risk above and a typo'd
///    `--fn`/`--ramp` reports its real cause instead of a generic probe
///    failure.
/// 2. **Spawn one child per probe.** The parent re-executes itself as
///    `soroban-testkit __limits-probe` (hidden from `--help`) with the
///    ramp value passed through `--value`. See [`run_probe`] for the child
///    half. The command creates no temporary Cargo projects and launches
///    no shell; the child's in-memory `Env` is disposable. `ProbeChild`
///    kills and reaps on ordinary interruption, error and early return.
/// 3. **Bound the probe.** The parent waits up to `--probe-timeout` in
///    [`wait_with_timeout`]. If the child finishes, its exit status decides
///    the outcome; if it overruns, the parent kills it and records a
///    [`ProbeFailureCategory::TimedOut`] whose stderr names the ramp value
///    and the timeout. A child terminated by a signal (the host abort above)
///    is recorded as [`ProbeFailureCategory::Abort`]. With `--timeout`, each
///    wait is also clamped to the time left before the overall deadline; a
///    probe cut short by that deadline fails the run with the best value
///    verified so far instead of being recorded as a ceiling.
/// 4. **Classify and search.** Every outcome is classified with
///    [`ProbeFailureCategory`]; the parent doubles to bracket the first
///    failure and then binary-searches with
///    [`refine_upper_bound_audited`], recording each probe's category.
/// 5. **Report once.** The collected results are assembled into a single
///    [`LimitsSummary`] that renders both the terminal output and the JSON
///    export, so the two can never disagree. It includes the last failing
///    probe ([`FailedProbe`]) next to the maximum success. With `--progress`,
///    one line per probe is also written to stderr as results come in.
///
/// If the parent cannot spawn a child at all, that is surfaced as a
/// [`ProbeFailureCategory::SpawnFailure`] with guidance to reinstall the
/// binary, rather than being mistaken for a contract ceiling.
///
/// # Scope
///
/// The ramp parameter must be a numeric type (`u32`/`i32`/`u64`/`i64`/
/// `u128`/`i128`, ramped as the value itself) or `Vec<T>` for a supported
/// element type `T` (ramped as the element *count*, each element filled
/// with a fixed value â see [`vec_element_val`]) â this covers the common
/// "maximum recipients in a batch operation" question directly, for
/// `Vec<Address>` and beyond. Every other parameter is filled with a fixed
/// default (an address, `0`, an empty collection, ...). A parameter type
/// this command doesn't know how to default (`Map`, `Tuple`, `Option`,
/// `Result`, a user-defined type, ...) is reported as an error rather than
/// guessed at.
///
/// Ledger read/write counts and transaction size are **not** reported:
/// they come from a transaction's simulated resource footprint, which
/// this command's native invocation does not produce (that requires the
/// network-dependent `stellar contract invoke` simulation path, which
/// this crate deliberately avoids â see its "zero network access"
/// constraint). Only CPU instructions and memory, available locally via
/// the host's budget, are reported.
pub fn run(args: LimitsArgs) -> Result<(), CliError> {
    validate_percentage("--baseline-tolerance-pct", args.baseline_tolerance_pct)?;
    validate_timeouts(args.probe_timeout, args.timeout)?;
    if args.jobs == 0 || args.jobs > MAX_PROBE_JOBS {
        return Err(CliError(format!(
            "--jobs must be between 1 and {MAX_PROBE_JOBS} (got {})",
            args.jobs
        )));
    }

    // Validate the configuration (function exists, ramp parameter exists,
    // every parameter's type is one this command knows how to default or
    // ramp) up front, in-process, before spawning any probes â this is
    // just type/spec checking, not invocation, so it carries none of the
    // abort risk documented above, and it means a typo'd --fn or --ramp
    // reports its actual cause instead of the generic "failed at value 1"
    // a swallowed child-process error would otherwise produce.
    let (function, ramp, effective_arg_values) = resolve_preset_defaults(&args)?;
    let (_, env, function_spec, ramp_index) = load(&args.contract, &function, &ramp)?;
    let co_ramp_indices: Vec<usize> = args
        .co_ramp
        .iter()
        .map(|name| {
            function_spec
                .inputs
                .iter()
                .position(|input| input.name.to_utf8_string_lossy() == name.as_str())
                .ok_or_else(|| {
                    CliError(format!(
                        "--co-ramp parameter {:?} not found in function {:?}",
                        name, function
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    build_args(
        &env,
        &function_spec,
        ramp_index,
        &co_ramp_indices,
        1,
        &effective_arg_values,
    )?;
    drop(env);

    if args.dry_run {
        println!("dry-run: would probe {:?} ramping {:?}", function, ramp);
        match args.search_strategy {
            SearchStrategy::Doubling => println!(
                "  phase 1: doubling ladder 1, 2, 4, 8, \u{2026} up to {SEARCH_UPPER_BOUND}"
            ),
            SearchStrategy::FixedStep => println!(
                "  phase 1: fixed-step ladder {step}, {two_step}, {three_step}, \u{2026} up to {SEARCH_UPPER_BOUND}",
                step = args.step,
                two_step = args.step.saturating_mul(2),
                three_step = args.step.saturating_mul(3),
            ),
        }
        println!("  phase 2: binary search to refine bounds");
        if !args.co_ramp.is_empty() {
            println!("  co-ramp parameters: {}", args.co_ramp.join(", "));
        }
        if args.max_probes > 0 {
            println!("  max probes: {}", args.max_probes);
        }
        return Ok(());
    }

    let self_exe = std::env::current_exe()
        .map_err(|err| CliError(format!("failed to locate this binary: {err}")))?;

    let probe_timeout = Duration::from_secs(args.probe_timeout);
    let clock = RunClock::start(args.timeout.map(Duration::from_secs));
    let progress = ProgressReporter::new(args.progress, &ramp, clock.started);
    let mut last_failed: Option<FailedProbe> = None;
    let mut probe_failure_counts: BTreeMap<ProbeFailureCategory, usize> = BTreeMap::new();
    let mut last_failed_probe: Option<u32> = None;
    let mut last_failure_category: Option<ProbeFailureCategory> = None;
    let mut first_failed_probe_stderr: Option<String> = None;
    let mut hit_upper_bound = false;
    let mut probe_count: usize = 0;
    let mut all_samples: Vec<ProbeSample> = Vec::new();

    // Folds one probe result into the run's statistics. Only called in ramp
    // order, so the counts and the first failure's stderr match a sequential run.
    // Highest success so far, shared with the deadline checks below.
    let best_so_far: std::cell::Cell<Option<u32>> = std::cell::Cell::new(None);
    let mut record = |phase: ProbePhase,
                      value: u32,
                      outcome: ProbeOutcome,
                      stderr: String,
                      duration_ms: Option<u64>| {
        let (succeeded, failure_category) = match &outcome {
            ProbeOutcome::Succeeded => (true, None),
            ProbeOutcome::Failed(cat) => (false, Some(*cat)),
        };
        all_samples.push(ProbeSample {
            value,
            succeeded,
            failure_category,
            duration_ms,
        });
        if succeeded {
            best_so_far.set(Some(
                best_so_far.get().map_or(value, |best| best.max(value)),
            ));
        }
        progress.report(all_samples.len(), phase, value, outcome, best_so_far.get());
        if let ProbeOutcome::Failed(category) = outcome {
            last_failed = Some(FailedProbe {
                value,
                category,
                stderr: Some(stderr.trim().to_string()).filter(|s| !s.is_empty()),
            });
            // A contract failure cannot establish a budget ceiling, even if
            // earlier, lower ramp values succeeded.
            if category == ProbeFailureCategory::ContractFailure {
                return Err(CliError(format!(
                    "contract function {:?} failed at {}={} before a resource limit was verified; \
                     check the contract's validation and trap logic; probe stderr: {}",
                    function,
                    ramp,
                    value,
                    stderr.trim()
                )));
            }
            *probe_failure_counts.entry(category).or_insert(0) += 1;
            if first_failed_probe_stderr.is_none() && !stderr.trim().is_empty() {
                first_failed_probe_stderr = Some(stderr);
            }
        }
        Ok(outcome)
    };

    // Validate fixed-step configuration before entering the scan loop.
    if args.search_strategy == SearchStrategy::FixedStep && args.step == 0 {
        return Err(CliError(
            "--step must be at least 1 when using --search-strategy fixed-step".to_string(),
        ));
    }

    // Find a failing upper bound. The strategy controls how the scan advances:
    // - Doubling: 1, 2, 4, 8, … (O(log N) probes; best for unknown ceilings)
    // - FixedStep: step, 2*step, 3*step, … (predictable sequence; uses --step)
    // Each batch holds up to `jobs` consecutive values probed concurrently;
    // results are read back in order and stop at the first failure.
    let mut last_ok: Option<u32> = None;
    let mut high = None;
    let mut next = Some(match args.search_strategy {
        SearchStrategy::Doubling => 1u32,
        SearchStrategy::FixedStep => args.step,
    });
    while let Some(start) = next {
        if args.max_probes > 0 && probe_count >= args.max_probes {
            break;
        }
        clock.check(probe_count, best_so_far.get(), &function, &ramp)?;
        let batch = match args.search_strategy {
            SearchStrategy::Doubling => ladder_batch(start, SEARCH_UPPER_BOUND, args.jobs),
            SearchStrategy::FixedStep => {
                fixed_step_batch(start, SEARCH_UPPER_BOUND, args.step, args.jobs)
            }
        };
        let wait = clock.probe_wait(probe_timeout);
        let results = probe_batch(&self_exe, &args, &batch, wait);
        for (value, result) in batch.into_iter().zip(results) {
            probe_count += 1;
            if args.max_probes > 0 && probe_count > args.max_probes {
                break;
            }
            let (outcome, stderr, duration_ms) = result?;
            clock.check_cut_short(
                outcome,
                wait,
                probe_timeout,
                value,
                probe_count,
                best_so_far.get(),
                &function,
                &ramp,
            )?;
            match record(ProbePhase::Scan, value, outcome, stderr, duration_ms)? {
                ProbeOutcome::Succeeded => {
                    last_ok = Some(value);
                    next = match args.search_strategy {
                        SearchStrategy::Doubling => next_ramp_value(value, SEARCH_UPPER_BOUND),
                        SearchStrategy::FixedStep => {
                            next_fixed_step_value(value, SEARCH_UPPER_BOUND, args.step)
                        }
                    };
                    if next.is_none() {
                        hit_upper_bound = true;
                    }
                }
                ProbeOutcome::Failed(category) => {
                    last_failed_probe = Some(value);
                    last_failure_category = Some(category);
                    high = Some(value);
                    next = None;
                    break;
                }
            }
        }
    }

    let Some(last_ok) = last_ok else {
        let category_note = match last_failure_category {
            Some(category) => format!(
                "; probe failure category: {category} ({})",
                category.guidance()
            ),
            None => String::new(),
        };
        let stderr_note = first_failed_probe_stderr
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| format!("; probe stderr: {s}"))
            .unwrap_or_default();
        return Err(CliError(format!(
            "{:?} failed even at the smallest ramp value (1) for parameter {:?}{category_note}\
             {stderr_note}; \
             this usually indicates a validation error in the contract rather than a \
             resource limit â check the contract's parameter validation logic and retry",
            function, ramp
        )));
    };

    // Binary search between the last success and the first failure for a
    // tighter bound, if we found one. Use the last failing probe value to
    // ensure we have a deterministic stopping condition.
    let mut best = last_ok;
    let mut first_failed = last_failed_probe;
    let mut first_failure_category = last_failure_category;
    let mut refinement_audit = None;
    if let Some(hi_start) = high {
        // Refinement is sequential: each midpoint depends on the previous result.
        let mut probe = |value: u32| -> Result<ProbeOutcome, CliError> {
            if args.max_probes > 0 && probe_count >= args.max_probes {
                return Ok(ProbeOutcome::Failed(ProbeFailureCategory::ExitFailure));
            }
            clock.check(probe_count, best_so_far.get(), &function, &ramp)?;
            probe_count += 1;
            let wait = clock.probe_wait(probe_timeout);
            let (outcome, stderr, duration_ms) = spawn_probe(&self_exe, &args, value, wait)?;
            clock.check_cut_short(
                outcome,
                wait,
                probe_timeout,
                value,
                probe_count,
                best_so_far.get(),
                &function,
                &ramp,
            )?;
            record(ProbePhase::Refine, value, outcome, stderr, duration_ms)
        };
        let (last_success, first_failure, audit) =
            refine_upper_bound_audited(last_ok, hi_start, &mut probe)?;
        best = last_success;
        first_failed = Some(first_failure);
        // The value that set the final upper bound is the last failing step,
        // so its category is the standardized reason for the boundary. Fall
        // back to the doubling probe's category if refinement took no steps.
        first_failure_category = audit
            .steps
            .iter()
            .rev()
            .find(|step| !step.succeeded)
            .and_then(|step| step.failure_category)
            .or(last_failure_category);
        refinement_audit = Some(audit);
    }

    // Measure resources at the best-known-good value, in-process this
    // time (a known-successful call is safe to make directly).
    let (instructions, memory_bytes) = measure_at(
        &args.contract,
        &function,
        &ramp,
        &effective_arg_values,
        best,
    )?;

    // Determine which resource boundary stopped the ramp.
    let stopped_by = if hit_upper_bound {
        Some(ResourceBoundary::Other)
    } else if first_failure_category == Some(ProbeFailureCategory::ContractFailure) {
        Some(ResourceBoundary::Other)
    } else {
        Some(infer_resource_boundary(instructions, memory_bytes))
    };

    let reproduction = Some(build_reproduction_guidance(
        &sanitize_report_path(&args.contract),
        &function,
        &ramp,
        best,
        &effective_arg_values,
    ));

    let summary = LimitsSummary {
        contract: sanitize_report_path(&args.contract),
        function: Some(function.clone()),
        ramp: Some(ramp.clone()),
        best,
        instructions,
        memory_bytes,
        first_failed,
        first_failure_category,
        first_failed_probe_stderr,
        last_failed_probe: last_failed,
        probe_failure_counts,
        hit_upper_bound,
        probe_timeout_secs: args.probe_timeout,
        audit: refinement_audit,
        measurement_limitations: MEASUREMENT_LIMITATIONS.to_string(),
        reproduction,
        stopped_by,
        probe_samples: all_samples,
        co_ramp: args.co_ramp.clone(),
        compiler_meta: Some(collect_compiler_meta()),
        wasm_target_dir: args
            .wasm_target_dir
            .as_ref()
            .map(|p| p.display().to_string()),
        profile: args.profile.clone(),
    };
    print!("{}", summary.render_text());

    if let Some(baseline_path) = &args.baseline {
        compare_to_baseline(
            baseline_path,
            &ramp,
            &function,
            best,
            instructions,
            memory_bytes,
            args.baseline_tolerance_pct,
        )?;
    }
    if let Some(save_path) = &args.save_baseline {
        write_baseline(
            save_path,
            &args.contract,
            &ramp,
            &function,
            best,
            instructions,
            memory_bytes,
        )?;
        println!("saved baseline to {}", save_path.display());
    }
    if let Some(export_path) = &args.export {
        export_results(export_path, &summary)?;
        println!("exported results to {}", export_path.display());
    }
    if let Some(budget_path) = &args.budget {
        compare_to_budget(budget_path, &ramp, best, instructions, memory_bytes)?;
    }
    if let Some(samples_path) = &args.export_samples {
        export_probe_samples(samples_path, &summary.probe_samples)?;
        println!("exported probe samples to {}", samples_path.display());
    }

    Ok(())
}

/// Runs one probe in its own process and returns its classified outcome with
/// the captured stderr and optional wall-clock duration. It touches no shared
/// state, so calls may run concurrently.
fn spawn_probe(
    self_exe: &Path,
    args: &LimitsArgs,
    value: u32,
    probe_timeout: Duration,
) -> Result<(ProbeOutcome, String, Option<u64>), CliError> {
    let probe_start = args.record_probe_duration.then(Instant::now);
    let child = Command::new(self_exe)
        .arg("__limits-probe")
        .arg("--contract")
        .arg(&args.contract)
        .arg("--fn")
        .arg(args.function.as_deref().unwrap_or(""))
        .arg("--ramp")
        .arg(args.ramp.as_deref().unwrap_or(""))
        .arg("--value")
        .arg(value.to_string())
        .args(
            args.arg_values
                .iter()
                .flat_map(|value| ["--arg", value.as_str()]),
        )
        .args(
            args.co_ramp
                .iter()
                .flat_map(|name| ["--co-ramp", name.as_str()]),
        )
        .args(
            args.wasm_target_dir
                .iter()
                .flat_map(|dir| ["--wasm-target-dir", dir.to_str().unwrap_or("")]),
        )
        .args(args.profile.iter().flat_map(|p| ["--profile", p.as_str()]))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|err| {
            CliError(format!(
                "failed to spawn the isolated probe process for ramp value {value}: {err}; \
                 verify the soroban-testkit binary is intact and re-run \
                 ({})",
                ProbeFailureCategory::SpawnFailure.guidance()
            ))
        })?;
    // Retain upstream's RAII cleanup on early returns and failures.
    let mut child = ProbeChild { child };

    let status = wait_with_timeout(&mut child.child, probe_timeout)?;
    let stderr = child
        .child
        .stderr
        .take()
        .map_or(Ok(String::new()), |mut stream| {
            use std::io::Read;
            let mut buffer = String::new();
            stream.read_to_string(&mut buffer).map(|_| buffer)
        })
        .map_err(|err| CliError(format!("failed to read probe stderr: {err}")))?;

    let outcome = match status {
        Some(status) if status.success() => ProbeOutcome::Succeeded,
        Some(status) if status.code().is_none() => {
            ProbeOutcome::Failed(ProbeFailureCategory::Abort)
        }
        Some(_) => ProbeOutcome::Failed(classify_probe_exit_stderr(&stderr)),
        None => ProbeOutcome::Failed(ProbeFailureCategory::TimedOut),
    };
    // A killed probe usually prints nothing, which would leave every report
    // with no explanation. Record which value hit which deadline.
    let stderr = if status.is_none() {
        let note = timed_out_note(args.ramp.as_deref().unwrap_or(""), value, probe_timeout);
        if stderr.trim().is_empty() {
            note
        } else {
            format!("{note}; partial stderr: {}", stderr.trim())
        }
    } else {
        stderr
    };
    let duration_ms = probe_start.map(|t| t.elapsed().as_millis() as u64);
    Ok((outcome, stderr, duration_ms))
}

/// Runs `values` as concurrent probes, one thread each, and returns the results
/// in the same order as `values`.
fn probe_batch(
    self_exe: &Path,
    args: &LimitsArgs,
    values: &[u32],
    probe_timeout: Duration,
) -> Vec<Result<(ProbeOutcome, String, Option<u64>), CliError>> {
    std::thread::scope(|scope| {
        let workers: Vec<_> = values
            .iter()
            .map(|&value| scope.spawn(move || spawn_probe(self_exe, args, value, probe_timeout)))
            .collect();
        workers
            .into_iter()
            .map(|worker| {
                worker.join().unwrap_or_else(|_| {
                    Err(CliError(
                        "an isolated probe worker panicked; re-run with --jobs 1 to isolate \
                         the failing probe"
                            .to_string(),
                    ))
                })
            })
            .collect()
    })
}

/// The next ramp values of the doubling ladder from `start`: at most `jobs`
/// values, each the successor of the one before, stopping at `upper_bound`.
fn ladder_batch(start: u32, upper_bound: u32, jobs: usize) -> Vec<u32> {
    let jobs = jobs.max(1);
    let mut batch = Vec::with_capacity(jobs);
    let mut current = Some(start);
    while let Some(value) = current {
        if batch.len() == jobs {
            break;
        }
        batch.push(value);
        current = next_ramp_value(value, upper_bound);
    }
    batch
}

/// Shell-quote a single argument for copyable POSIX-shell guidance.
/// Actual probes use Command::arg directly and never launch a shell or Cargo.
fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_./:=+".contains(&byte))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// Builds shell-safe instructions for reproducing the discovered ceiling.
/// Spaces and single quotes in the WASM path remain one argument.
pub fn build_reproduction_guidance(
    contract: &str,
    function: &str,
    ramp: &str,
    best: u32,
    arg_values: &[String],
) -> String {
    let mut cmd = format!(
        "soroban-testkit limits --contract {} --fn {} --ramp {}",
        shell_quote(contract),
        shell_quote(function),
        shell_quote(ramp)
    );
    for arg in arg_values {
        cmd.push_str(&format!(" --arg {}", shell_quote(arg)));
    }
    format!(
        "invoke `{function}` with `{ramp}` = {best} in a test environment; to re-verify the boundary via CLI run: `{cmd}`"
    )
}

/// Context recorded for a probe the parent killed at its deadline.
fn timed_out_note(ramp: &str, value: u32, timeout: Duration) -> String {
    format!(
        "probe for {ramp}={value} was killed after exceeding its {} timeout; \
         raise --probe-timeout if this value should finish",
        format_duration(timeout)
    )
}

/// `1.5s`, `30s`, `250ms`: short, unit-bearing durations for messages.
fn format_duration(duration: Duration) -> String {
    let ms = duration.as_millis();
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms.is_multiple_of(1_000) {
        format!("{}s", ms / 1_000)
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

/// Rejects timeout settings the run could never honour, before any probe is
/// spawned, so a bad flag is reported as such rather than as a probe failure.
fn validate_timeouts(probe_timeout_secs: u64, timeout_secs: Option<u64>) -> Result<(), CliError> {
    let fits_clock = |secs: u64| {
        Instant::now()
            .checked_add(Duration::from_secs(secs))
            .is_some()
    };
    if !fits_clock(probe_timeout_secs) {
        return Err(CliError(format!(
            "--probe-timeout {probe_timeout_secs} is too large for the host clock; \
             use a realistic per-probe limit such as 30 or 300 seconds"
        )));
    }
    if let Some(total) = timeout_secs {
        if total == 0 {
            return Err(CliError(
                "--timeout must be at least 1 second; omit it to let the command run until the \
                 search finishes"
                    .to_string(),
            ));
        }
        if !fits_clock(total) {
            return Err(CliError(format!(
                "--timeout {total} is too large for the host clock; omit it to run without an \
                 overall deadline"
            )));
        }
    }
    Ok(())
}

/// Which search phase a probe belongs to, for progress output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbePhase {
    /// Doubling / fixed-step scan that brackets the first failure.
    Scan,
    /// Binary refinement between the last success and the first failure.
    Refine,
}

impl ProbePhase {
    fn label(self) -> &'static str {
        match self {
            ProbePhase::Scan => "scan",
            ProbePhase::Refine => "refine",
        }
    }
}

/// Formats one `--progress` line (issue #362).
fn format_progress_line(
    probe_number: usize,
    phase: ProbePhase,
    ramp: &str,
    value: u32,
    outcome: ProbeOutcome,
    best_so_far: Option<u32>,
    elapsed: Duration,
) -> String {
    let result = match outcome {
        ProbeOutcome::Succeeded => "ok".to_string(),
        ProbeOutcome::Failed(category) => format!("failed [{category}]"),
    };
    let best = best_so_far.map_or_else(|| "none yet".to_string(), |best| best.to_string());
    format!(
        "limits: probe {probe_number} ({phase}) {ramp}={value} {result}; best so far: {best}; \
         elapsed {}",
        format_duration(elapsed),
        phase = phase.label(),
    )
}

/// Writes `--progress` lines to stderr when enabled; a no-op otherwise.
struct ProgressReporter<'a> {
    enabled: bool,
    ramp: &'a str,
    started: Instant,
}

impl<'a> ProgressReporter<'a> {
    fn new(enabled: bool, ramp: &'a str, started: Instant) -> Self {
        ProgressReporter {
            enabled,
            ramp,
            started,
        }
    }

    fn report(
        &self,
        probe_number: usize,
        phase: ProbePhase,
        value: u32,
        outcome: ProbeOutcome,
        best_so_far: Option<u32>,
    ) {
        if self.enabled {
            eprintln!(
                "{}",
                format_progress_line(
                    probe_number,
                    phase,
                    self.ramp,
                    value,
                    outcome,
                    best_so_far,
                    self.started.elapsed(),
                )
            );
        }
    }
}

/// Tracks the optional overall `--timeout` deadline (issue #361).
struct RunClock {
    started: Instant,
    limit: Option<Duration>,
    deadline: Option<Instant>,
}

impl RunClock {
    fn start(limit: Option<Duration>) -> Self {
        let started = Instant::now();
        // validate_timeouts already rejected limits that overflow the clock.
        let deadline = limit.and_then(|limit| started.checked_add(limit));
        RunClock {
            started,
            limit,
            deadline,
        }
    }

    /// Time left before the overall deadline; `None` when there is none.
    fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// How long the next probe may run: the per-probe timeout, clamped to the
    /// time left before the overall deadline.
    fn probe_wait(&self, probe_timeout: Duration) -> Duration {
        match self.remaining() {
            Some(remaining) => probe_timeout.min(remaining),
            None => probe_timeout,
        }
    }

    /// Errors once the overall deadline has passed.
    fn check(
        &self,
        probes_run: usize,
        best: Option<u32>,
        function: &str,
        ramp: &str,
    ) -> Result<(), CliError> {
        match self.remaining() {
            Some(remaining) if remaining.is_zero() => {
                Err(self.exceeded(probes_run, best, None, function, ramp))
            }
            _ => Ok(()),
        }
    }

    /// A probe that timed out only because the overall deadline clamped its
    /// wait says nothing about the contract, so it ends the run instead of
    /// being recorded as a `timed-out` ceiling.
    #[allow(clippy::too_many_arguments)]
    fn check_cut_short(
        &self,
        outcome: ProbeOutcome,
        wait: Duration,
        probe_timeout: Duration,
        value: u32,
        probes_run: usize,
        best: Option<u32>,
        function: &str,
        ramp: &str,
    ) -> Result<(), CliError> {
        let cut_short =
            outcome == ProbeOutcome::Failed(ProbeFailureCategory::TimedOut) && wait < probe_timeout;
        if cut_short {
            Err(self.exceeded(probes_run, best, Some(value), function, ramp))
        } else {
            Ok(())
        }
    }

    fn exceeded(
        &self,
        probes_run: usize,
        best: Option<u32>,
        interrupted: Option<u32>,
        function: &str,
        ramp: &str,
    ) -> CliError {
        let limit = self.limit.map(format_duration).unwrap_or_default();
        let best = match best {
            Some(best) => format!("the highest {ramp} verified so far is {best}"),
            None => format!("no {ramp} value had succeeded yet"),
        };
        let interrupted = interrupted
            .map(|value| format!(" while probing {ramp}={value}"))
            .unwrap_or_default();
        CliError(format!(
            "limits for {function:?} exceeded --timeout ({limit}) after {probes_run} probe(s)\
             {interrupted}; {best}, but the boundary was not confirmed. Raise --timeout, lower \
             --probe-timeout, or use --jobs to probe faster, then re-run"
        ))
    }
}

fn validate_percentage(name: &str, pct: f64) -> Result<(), CliError> {
    if !pct.is_finite() || !(0.0..=100.0).contains(&pct) {
        return Err(CliError(format!(
            "{name} must be between 0 and 100 (got {pct})"
        )));
    }
    Ok(())
}

// Stop at the cap and clamp the last step, including when doubling would overflow.
fn next_ramp_value(current: u32, upper_bound: u32) -> Option<u32> {
    if current >= upper_bound {
        None
    } else {
        Some(current.saturating_mul(2).min(upper_bound))
    }
}

/// Next value in a fixed-step sequence: `current + step`, clamped to `upper_bound`.
/// Returns `None` when `current` has already reached the bound.
fn next_fixed_step_value(current: u32, upper_bound: u32, step: u32) -> Option<u32> {
    if current >= upper_bound {
        None
    } else {
        Some(current.saturating_add(step).min(upper_bound))
    }
}

/// A batch of up to `jobs` fixed-step values starting at `start`, each `step`
/// apart, stopping at `upper_bound`.
fn fixed_step_batch(start: u32, upper_bound: u32, step: u32, jobs: usize) -> Vec<u32> {
    let jobs = jobs.max(1);
    let step = step.max(1);
    let mut batch = Vec::with_capacity(jobs);
    let mut current = Some(start.min(upper_bound));
    while let Some(value) = current {
        if batch.len() == jobs {
            break;
        }
        batch.push(value);
        current = next_fixed_step_value(value, upper_bound, step);
    }
    batch
}

/// Refines the boundary between a known success (`low`) and a known failure (`high`)
/// using binary search, recording every intermediate step in a [`RefinementAudit`]
/// â including the standardized [`ProbeFailureCategory`] of each failing step.
/// Re-probes both endpoints before refinement and rejects a contradictory bracket.
/// The contract must succeed monotonically up to its ceiling: these checks detect
/// observed inconsistencies, but cannot prove monotonicity at unprobed values.
pub fn refine_upper_bound_audited<F>(
    mut low: u32,
    mut high: u32,
    probe: &mut F,
) -> Result<(u32, u32, RefinementAudit), CliError>
where
    F: FnMut(u32) -> Result<ProbeOutcome, CliError>,
{
    if low >= high {
        return Err(CliError(format!(
            "refine_upper_bound requires lower bound ({low}) < upper bound ({high})"
        )));
    }

    if !probe(low)?.succeeded() {
        return Err(CliError(format!(
            "non-monotonic probe results: lower bound {low} no longer succeeds; \
             binary refinement requires a stable successful lower bound and failing upper bound; \
             check contract validation and probe timeouts, then re-run"
        )));
    }
    if probe(high)?.succeeded() {
        return Err(CliError(format!(
            "non-monotonic probe results: upper bound {high} now succeeds; \
             binary refinement requires a stable successful lower bound and failing upper bound; \
             check contract validation and probe timeouts, then re-run"
        )));
    }

    let initial_low = low;
    let initial_high = high;
    let mut steps = Vec::new();
    let mut step_num = 1;

    while low.saturating_add(1) < high {
        let middle = low + (high - low) / 2;
        let outcome = probe(middle)?;
        let succeeded = outcome.succeeded();
        let failure_category = match outcome {
            ProbeOutcome::Succeeded => None,
            ProbeOutcome::Failed(category) => Some(category),
        };
        if succeeded {
            low = middle;
        } else {
            high = middle;
        }
        steps.push(RefinementStep {
            step: step_num,
            probe_value: middle,
            succeeded,
            failure_category,
            window_low: low,
            window_high: high,
        });
        step_num += 1;
    }

    let audit = RefinementAudit {
        initial_low,
        initial_high,
        steps,
        final_success: low,
        final_failure: high,
    };

    Ok((low, high, audit))
}

/// Bool-returning convenience wrapper around [`refine_upper_bound_audited`],
/// retained for callers (and tests) that only care about success/failure.
/// A `false` probe is classified as [`ProbeFailureCategory::ExitFailure`],
/// the category the isolated probe reports for a clean non-zero exit.
#[allow(dead_code)]
pub fn refine_upper_bound<F>(low: u32, high: u32, probe: &mut F) -> Result<(u32, u32), CliError>
where
    F: FnMut(u32) -> Result<bool, CliError>,
{
    let (success, failure, _) = refine_upper_bound_audited(low, high, &mut |value| {
        Ok(if probe(value)? {
            ProbeOutcome::Succeeded
        } else {
            ProbeOutcome::Failed(ProbeFailureCategory::ExitFailure)
        })
    })?;
    Ok((success, failure))
}

/// The hidden child half of the isolation lifecycle documented on [`run`]:
/// performs exactly one invocation in a throwaway [`Env`] registered from
/// the `.wasm` and returns `Ok(())` (exit 0) or `Err` (exit 1) â no
/// measurement, no output. Because the child dies after a single call, the
/// parent can classify an abort by signal without losing its own run.
///
/// The child only ever reports success/failure through its exit status; the
/// parent turns that status into a [`ProbeFailureCategory`]. Failures that
/// happen *before* invocation (an unreadable contract, a bad ramp value)
/// come back as an actionable [`CliError`] message, but the parent's
/// pre-validation normally catches those first.
pub fn run_probe(args: ProbeArgs) -> Result<(), CliError> {
    let _quiet = QuietPanics::install();
    let (contract_id, env, function, ramp_index) =
        load(&args.contract, &args.function, &args.ramp)?;
    let co_ramp_indices: Vec<usize> = args
        .co_ramp
        .iter()
        .filter_map(|name| {
            function
                .inputs
                .iter()
                .position(|input| input.name.to_utf8_string_lossy() == name.as_str())
        })
        .collect();
    let args_vec = build_args(
        &env,
        &function,
        ramp_index,
        &co_ramp_indices,
        args.value,
        &args.arg_values,
    )?;
    let func = Symbol::new(&env, &args.function);
    match env.try_invoke_contract::<Val, soroban_sdk::Error>(&contract_id, &func, args_vec) {
        Ok(Ok(_)) => Ok(()),
        Err(Ok(err)) if !err.is_type(soroban_sdk::xdr::ScErrorType::Budget) => Err(CliError(
            format!("contract-failure: invocation returned {err:?}"),
        )),
        _ => Err(CliError("probe failed".to_string())),
    }
}

/// Owns one isolated probe. On ordinary error, early return or unwinding,
/// terminate and reap any child still running. No temporary Cargo project or
/// source directory is created by this command. An uncatchable OS termination
/// of the parent cannot invoke Drop.
struct ProbeChild {
    child: Child,
}

impl Drop for ProbeChild {
    fn drop(&mut self) {
        if let Ok(Some(_)) = self.child.try_wait() {
            return;
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Poll one child until it exits or exceeds its deadline. A timeout kills
/// and reaps before returning None. Errors name the process rather than
/// silently claiming a resource ceiling. ProbeChild handles early errors.
fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Result<Option<ExitStatus>, CliError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| CliError("probe timeout is too large for the host clock".to_string()))?;
    loop {
        if let Some(status) = child.try_wait().map_err(|err| {
            CliError(format!(
                "failed to poll isolated probe process {}: {err}",
                child.id()
            ))
        })? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            // A child can exit between polling and kill. In that race,
            // try_wait reaps it; otherwise report the failed termination.
            match child.kill() {
                Ok(()) => {
                    child.wait().map_err(|err| {
                        CliError(format!(
                            "failed to reap timed-out probe process {}: {err}",
                            child.id()
                        ))
                    })?;
                }
                Err(kill_err) => match child.try_wait() {
                    Ok(Some(_)) => {} // Already exited and reaped.
                    Ok(None) => {
                        return Err(CliError(format!(
                            "failed to terminate timed-out probe process {}: {kill_err}",
                            child.id()
                        )));
                    }
                    Err(poll_err) => {
                        return Err(CliError(format!(
                            "failed to terminate or poll timed-out probe process {}: \
                             kill: {kill_err}; poll: {poll_err}",
                            child.id()
                        )));
                    }
                },
            }
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Sanitizes a filesystem path for machine-readable reports, ensuring host-specific
/// or process-specific temporary directory prefixes do not leak into reproducible outputs.
pub fn sanitize_report_path(path: &Path) -> String {
    let temp_dir = std::env::temp_dir();
    let is_temp = path.starts_with(&temp_dir)
        || temp_dir
            .canonicalize()
            .map(|t| path.starts_with(t))
            .unwrap_or(false)
        || path.to_string_lossy().starts_with("/tmp")
        || path.to_string_lossy().starts_with("/var/tmp");

    if is_temp {
        return path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "contract.wasm".to_string());
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(rel) = path.strip_prefix(&cwd) {
            return rel.to_string_lossy().to_string();
        }
        if let Ok(canonical_cwd) = cwd.canonicalize() {
            if let Ok(rel) = path.strip_prefix(&canonical_cwd) {
                return rel.to_string_lossy().to_string();
            }
        }
    }

    path.to_string_lossy().to_string()
}

/// A previously-saved `limits` result, for `--baseline` regression checks.
/// Stored as a flat `key=value` file recording ceiling and resource metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Baseline {
    pub contract: Option<String>,
    pub function: String,
    pub ramp: String,
    pub best: u32,
    pub instructions: u64,
    pub memory_bytes: u64,
    pub measurement_limitations: Option<String>,
}

impl Baseline {
    pub fn to_file_contents(&self) -> String {
        let mut out = String::new();
        if let Some(c) = &self.contract {
            out.push_str(&format!("contract={c}\n"));
        }
        out.push_str(&format!(
            "function={}\nramp={}\nbest={}\ninstructions={}\nmemory_bytes={}\nmeasurement_limitations={}\n",
            self.function,
            self.ramp,
            self.best,
            self.instructions,
            self.memory_bytes,
            self.measurement_limitations
                .as_deref()
                .unwrap_or(MEASUREMENT_LIMITATIONS),
        ));
        out
    }

    pub fn parse(contents: &str) -> Result<Self, CliError> {
        let mut contract = None;
        let mut function = None;
        let mut ramp = None;
        let mut best = None;
        let mut instructions = None;
        let mut memory_bytes = None;
        let mut measurement_limitations = None;

        for (line_no, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(CliError(format!(
                    "malformed baseline file at line {}: expected key=value, got {line:?}",
                    line_no + 1
                )));
            };
            match key {
                "contract" => contract = Some(value.to_string()),
                "function" => function = Some(value.to_string()),
                "ramp" => ramp = Some(value.to_string()),
                "best" => {
                    best = Some(value.parse::<u32>().map_err(|err| {
                        CliError(format!("malformed baseline `best` value {value:?}: {err}"))
                    })?)
                }
                "instructions" => {
                    instructions = Some(value.parse::<u64>().map_err(|err| {
                        CliError(format!(
                            "malformed baseline `instructions` value {value:?}: {err}"
                        ))
                    })?)
                }
                "memory_bytes" => {
                    memory_bytes = Some(value.parse::<u64>().map_err(|err| {
                        CliError(format!(
                            "malformed baseline `memory_bytes` value {value:?}: {err}"
                        ))
                    })?)
                }
                "measurement_limitations" => measurement_limitations = Some(value.to_string()),
                other => {
                    return Err(CliError(format!(
                        "malformed baseline file at line {}: unknown key {other:?}",
                        line_no + 1
                    )))
                }
            }
        }

        Ok(Baseline {
            contract,
            function: function
                .ok_or_else(|| CliError("baseline file is missing `function`".to_string()))?,
            ramp: ramp.ok_or_else(|| CliError("baseline file is missing `ramp`".to_string()))?,
            best: best.ok_or_else(|| CliError("baseline file is missing `best`".to_string()))?,
            instructions: instructions
                .ok_or_else(|| CliError("baseline file is missing `instructions`".to_string()))?,
            memory_bytes: memory_bytes
                .ok_or_else(|| CliError("baseline file is missing `memory_bytes`".to_string()))?,
            measurement_limitations: measurement_limitations
                .or_else(|| Some(MEASUREMENT_LIMITATIONS.to_string())),
        })
    }
}

pub fn write_baseline(
    path: &Path,
    contract: &Path,
    ramp: &str,
    function: &str,
    best: u32,
    instructions: u64,
    memory_bytes: u64,
) -> Result<(), CliError> {
    let sanitized_contract = sanitize_report_path(contract);
    let baseline = Baseline {
        contract: Some(sanitized_contract),
        function: function.to_string(),
        ramp: ramp.to_string(),
        best,
        instructions,
        memory_bytes,
        measurement_limitations: Some(MEASUREMENT_LIMITATIONS.to_string()),
    };
    std::fs::write(path, baseline.to_file_contents()).map_err(|err| {
        CliError(format!(
            "failed to write baseline to {}: {err}",
            path.display()
        ))
    })
}

/// Compares this run's result against the baseline saved at `path`, printing
/// a before/after summary and returning an error (which `main` turns into a
/// non-zero exit) if it regressed beyond `--baseline-tolerance-pct`.
fn compare_to_baseline(
    path: &std::path::Path,
    ramp: &str,
    function: &str,
    best: u32,
    instructions: u64,
    memory_bytes: u64,
    tolerance_pct: f64,
) -> Result<(), CliError> {
    let contents = std::fs::read_to_string(path).map_err(|err| {
        CliError(format!(
            "failed to read baseline at {}: {err}",
            path.display()
        ))
    })?;
    let baseline = Baseline::parse(&contents)?;

    if baseline.function != function || baseline.ramp != ramp {
        return Err(CliError(format!(
            "baseline at {} was recorded for {:?}/{:?}, not {function:?}/{ramp:?}; comparing \
             across different functions or ramp parameters isn't meaningful",
            path.display(),
            baseline.function,
            baseline.ramp
        )));
    }

    println!("baseline comparison ({}):", path.display());
    println!("  {ramp}: {} -> {best}", baseline.best);
    println!(
        "  instructions: {} -> {instructions}",
        baseline.instructions
    );
    println!(
        "  memory bytes: {} -> {memory_bytes}",
        baseline.memory_bytes
    );

    let mut regressions = Vec::new();
    if best < baseline.best {
        regressions.push(format!(
            "{ramp} ceiling dropped from {} to {best}",
            baseline.best
        ));
    }
    check_metric_regression(
        "instructions",
        baseline.instructions,
        instructions,
        tolerance_pct,
        &mut regressions,
    );
    check_metric_regression(
        "memory bytes",
        baseline.memory_bytes,
        memory_bytes,
        tolerance_pct,
        &mut regressions,
    );

    fn check_metric_regression(
        name: &str,
        old: u64,
        new: u64,
        tolerance_pct: f64,
        out: &mut Vec<String>,
    ) {
        if new <= old {
            return;
        }
        let growth_pct = ((new - old) as f64 / old.max(1) as f64) * 100.0;
        if growth_pct <= tolerance_pct {
            return;
        }
        out.push(format!(
            "{name} grew from {old} to {new} ({growth_pct:.1}%, over the {tolerance_pct:.1}% tolerance)"
        ));
    }

    if regressions.is_empty() {
        println!("  no regression");
        return Ok(());
    }

    // Instruction/memory growth within tolerance is reported above but not
    // treated as a regression; only the ramp-ceiling check and
    // over-tolerance growth fail the run.
    Err(CliError(format!(
        "baseline regression detected: {}",
        regressions.join("; ")
    )))
}

/// Writes `summary` to `path` in the format implied by its extension:
/// pretty JSON for `.json`, otherwise the flat `key=value` baseline format.
///
/// Both branches are rendered from the same [`LimitsSummary`] the terminal
/// output uses, so a boundary or failure category can never differ between
/// what a human sees and what CI reads back.
pub fn export_results(path: &Path, summary: &LimitsSummary) -> Result<(), CliError> {
    if path.extension().is_some_and(|ext| ext == "json") {
        let json_str = serde_json::to_string_pretty(summary)
            .map_err(|err| CliError(format!("failed to format JSON export: {err}")))?;
        std::fs::write(path, json_str).map_err(|err| {
            CliError(format!(
                "failed to export results to {}: {err}",
                path.display()
            ))
        })
    } else {
        let baseline = Baseline {
            contract: Some(summary.contract.clone()),
            function: summary.function.clone().unwrap_or_default(),
            ramp: summary.ramp.clone().unwrap_or_default(),
            best: summary.best,
            instructions: summary.instructions,
            memory_bytes: summary.memory_bytes,
            measurement_limitations: Some(summary.measurement_limitations.clone()),
        };
        std::fs::write(path, baseline.to_file_contents()).map_err(|err| {
            CliError(format!(
                "failed to export results to {}: {err}",
                path.display()
            ))
        })
    }
}

/// Writes all probe samples collected during the run to `path` as a JSON array.
pub fn export_probe_samples(path: &Path, samples: &[ProbeSample]) -> Result<(), CliError> {
    let json_str = serde_json::to_string_pretty(samples)
        .map_err(|err| CliError(format!("failed to format probe samples JSON: {err}")))?;
    std::fs::write(path, json_str).map_err(|err| {
        CliError(format!(
            "failed to write probe samples to {}: {err}",
            path.display()
        ))
    })
}

fn measure_at(
    contract: &std::path::Path,
    function: &str,
    ramp: &str,
    arg_values: &[String],
    value: u32,
) -> Result<(u64, u64), CliError> {
    let (contract_id, env, fn_spec, ramp_index) = load(contract, function, ramp)?;
    let func = Symbol::new(&env, function);
    let mut budget = env.cost_estimate().budget();
    budget.reset_default();
    let args_vec = build_args(&env, &fn_spec, ramp_index, &[], value, arg_values)?;
    if !invoke_succeeds(&env, &contract_id, &func, args_vec) {
        return Err(CliError(format!(
            "internal error: {:?} reported ramp value {value} as successful during probing              but failed on in-process re-measurement for parameter {:?}; re-run to confirm the              boundary is stable, or lower --probe-timeout if the run is racing a hang",
            function, ramp
        )));
    }
    Ok((budget.cpu_instruction_cost(), budget.memory_bytes_cost()))
}

/// Infer which Soroban resource boundary was most likely the binding constraint.
///
/// Compares each metric as a fraction of its known mainnet limit. The resource
/// with the higher fraction is returned; below a 25% threshold, [`ResourceBoundary::Other`]
/// is returned because neither resource was notably consumed.
fn infer_resource_boundary(instructions: u64, memory_bytes: u64) -> ResourceBoundary {
    let insn_frac = instructions as f64 / SOROBAN_CPU_INSN_LIMIT as f64;
    let mem_frac = memory_bytes as f64 / SOROBAN_MEMORY_LIMIT as f64;
    const THRESHOLD: f64 = 0.25;
    if insn_frac < THRESHOLD && mem_frac < THRESHOLD {
        ResourceBoundary::Other
    } else if insn_frac >= mem_frac {
        ResourceBoundary::Instructions
    } else {
        ResourceBoundary::Memory
    }
}

/// Resolve --fn, --ramp, and --arg values by applying any --preset defaults.
///
/// Explicit --fn/--ramp/--arg flags always win over preset defaults.
/// Returns an error if --fn or --ramp are absent and no preset is given.
fn resolve_preset_defaults(args: &LimitsArgs) -> Result<(String, String, Vec<String>), CliError> {
    let (function, ramp, extra_args): (String, String, &[&str]) =
        if let Some(preset_name) = &args.preset {
            let preset = RampPreset::from_name(preset_name)?;
            let function = args
                .function
                .clone()
                .unwrap_or_else(|| preset.function().to_string());
            let ramp = args
                .ramp
                .clone()
                .unwrap_or_else(|| preset.ramp().to_string());
            (function, ramp, preset.default_args())
        } else {
            let function = args
                .function
                .clone()
                .ok_or_else(|| CliError("--fn is required unless --preset is given".to_string()))?;
            let ramp = args.ramp.clone().ok_or_else(|| {
                CliError("--ramp is required unless --preset is given".to_string())
            })?;
            (function, ramp, &[])
        };

    let mut effective_args = args.arg_values.clone();
    for default_arg in extra_args {
        let arg_name = default_arg.split('=').next().unwrap_or("");
        if !effective_args
            .iter()
            .any(|a| a.starts_with(&format!("{arg_name}=")))
        {
            effective_args.push(default_arg.to_string());
        }
    }

    Ok((function, ramp, effective_args))
}

/// Compare a run's results against a checked-in resource budget file.
///
/// Prints a before/after summary for each checked field and returns an error
/// if any limit is exceeded, so CI fails actionably rather than silently.
fn compare_to_budget(
    path: &std::path::Path,
    ramp: &str,
    best: u32,
    instructions: u64,
    memory_bytes: u64,
) -> Result<(), CliError> {
    let contents = std::fs::read_to_string(path).map_err(|err| {
        CliError(format!(
            "failed to read resource budget at {}: {err}",
            path.display()
        ))
    })?;
    let budget = ResourceBudget::parse(&contents)?;

    println!("resource budget check ({}):", path.display());

    let mut violations = Vec::new();

    if let Some(min_best) = budget.min_best {
        println!("  {ramp} ceiling: {best} (budget minimum: {min_best})");
        if best < min_best {
            violations.push(format!(
                "{ramp} ceiling {best} is below the budget minimum of {min_best}"
            ));
        }
    }
    if let Some(max_insn) = budget.max_instructions {
        println!("  instructions: {instructions} (budget maximum: {max_insn})");
        if instructions > max_insn {
            violations.push(format!(
                "instructions {instructions} exceed the budget maximum of {max_insn}"
            ));
        }
    }
    if let Some(max_mem) = budget.max_memory_bytes {
        println!("  memory bytes: {memory_bytes} (budget maximum: {max_mem})");
        if memory_bytes > max_mem {
            violations.push(format!(
                "memory bytes {memory_bytes} exceed the budget maximum of {max_mem}"
            ));
        }
    }

    if violations.is_empty() {
        println!("  all checks passed");
        Ok(())
    } else {
        Err(CliError(format!(
            "resource budget exceeded: {}",
            violations.join("; ")
        )))
    }
}

/// Silences the default panic hook for the duration it's held, restoring
/// the previous hook on drop, so a probe's failure doesn't dump a panic
/// message and the host's diagnostic event log to stderr.
type PanicHook = Box<dyn Fn(&panic::PanicHookInfo<'_>) + Sync + Send>;

struct QuietPanics {
    previous: Option<PanicHook>,
}

impl QuietPanics {
    fn install() -> Self {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        Self {
            previous: Some(previous),
        }
    }
}

impl Drop for QuietPanics {
    fn drop(&mut self) {
        if let Some(hook) = self.previous.take() {
            panic::set_hook(hook);
        }
    }
}

fn load(
    contract: &std::path::Path,
    function: &str,
    ramp: &str,
) -> Result<(Address, Env, ScSpecFunctionV0, usize), CliError> {
    let wasm = std::fs::read(contract).map_err(|err| {
        let guidance = if err.kind() == std::io::ErrorKind::NotFound {
            "contract file not found; check --contract (relative paths use the current directory) \
             and build the contract successfully before retrying"
        } else {
            "check that --contract points to a readable .wasm file"
        };
        CliError(format!(
            "failed to read {}: {err}; {guidance}",
            contract.display()
        ))
    })?;

    let entries = soroban_spec::read::from_wasm(&wasm)
        .map_err(|err| CliError(format!("failed to read contract spec: {err}")))?;

    if entries.is_empty() {
        return Err(CliError(format!(
            "{}: contract has no exported functions in its spec; \
             did you include #[contract] and #[contractimpl] on your contract?",
            contract.display()
        )));
    }

    let available_functions: Vec<String> = entries
        .iter()
        .filter_map(|entry| match entry {
            ScSpecEntry::FunctionV0(f) => Some(f.name.to_string()),
            _ => None,
        })
        .collect();

    let func_spec = entries
        .into_iter()
        .find_map(|entry| match entry {
            ScSpecEntry::FunctionV0(f) if f.name.to_string() == function => Some(f),
            _ => None,
        })
        .ok_or_else(|| {
            if available_functions.is_empty() {
                CliError(format!(
                    "{}: contract has no exported functions in its spec; \
                     did you include #[contract] and #[contractimpl] on your contract?",
                    contract.display()
                ))
            } else {
                CliError(format!(
                    "no function named {function:?} in {}'s spec; \
                     available functions: {}",
                    contract.display(),
                    available_functions.join(", ")
                ))
            }
        })?;

    let ramp_index = func_spec
        .inputs
        .iter()
        .position(|input| input.name.to_utf8_string_lossy() == ramp)
        .ok_or_else(|| {
            let param_names: Vec<String> = func_spec
                .inputs
                .iter()
                .map(|p| p.name.to_utf8_string_lossy().to_string())
                .collect();
            let (available, guidance) = if param_names.is_empty() {
                (
                    "(none)".to_string(),
                    "choose a function with a numeric, Bytes, or Vec parameter using --fn",
                )
            } else {
                (
                    param_names.join(", "),
                    "set --ramp to an input name from the selected function's contract ABI",
                )
            };
            CliError(format!(
                "{function:?} has no parameter named {ramp:?}; \
                 available parameters: {available}; {guidance}"
            ))
        })?;

    let env = Env::new_with_config(soroban_sdk::testutils::EnvTestConfig {
        capture_snapshot_at_drop: false,
    });
    // Auth is not what this command measures â it exists to find a
    // resource ceiling, not to prove who may call the function â and each
    // probe runs in its own throwaway process/Env, so blanket-approving
    // auth here carries none of the AuthMatrix-correctness risk that a
    // shared, longer-lived TestEnv would (see TestToken's mint, which
    // deliberately does NOT do this for that reason).
    env.mock_all_auths_allowing_non_root_auth();
    let contract_id = env.register(wasm.as_slice(), ());

    Ok((contract_id, env, func_spec, ramp_index))
}

fn build_args(
    env: &Env,
    function: &ScSpecFunctionV0,
    ramp_index: usize,
    co_ramp_indices: &[usize],
    ramp_value: u32,
    explicit: &[String],
) -> Result<SVec<Val>, CliError> {
    let mut overrides = std::collections::HashMap::new();
    for item in explicit {
        let (name, value) = item
            .split_once('=')
            .ok_or_else(|| CliError(format!("invalid --arg {item:?}; expected NAME=VALUE")))?;
        if name.is_empty()
            || overrides
                .insert(name.to_string(), value.to_string())
                .is_some()
        {
            return Err(CliError(format!(
                "empty or duplicate --arg parameter {name:?}"
            )));
        }
    }
    let admin = Address::generate(env);
    let mut vals = SVec::new(env);
    let ramp_name = function.inputs[ramp_index].name.to_utf8_string_lossy();
    for name in overrides.keys() {
        if name == &ramp_name {
            return Err(CliError(format!(
                "--arg cannot override ramp parameter {name:?}"
            )));
        }
        if !function
            .inputs
            .iter()
            .any(|input| input.name.to_utf8_string_lossy() == *name)
        {
            return Err(CliError(format!("--arg names unknown parameter {name:?}")));
        }
    }
    for (i, input) in function.inputs.iter().enumerate() {
        let input_name = input.name.to_utf8_string_lossy();
        let val = if i == ramp_index || co_ramp_indices.contains(&i) {
            ramp_val(env, &input.type_, ramp_value)?
        } else if let Some(value) = overrides.get(input_name.as_str()) {
            explicit_val(env, &input.type_, value)?
        } else {
            default_val(env, &admin, &input.type_, input)?
        };
        vals.push_back(val);
    }
    Ok(vals)
}

fn explicit_val(env: &Env, type_: &ScSpecTypeDef, value: &str) -> Result<Val, CliError> {
    macro_rules! parse {
        ($ty:ty) => {
            value
                .parse::<$ty>()
                .map_err(|err| CliError(format!("invalid value {value:?}: {err}")))?
                .into_val(env)
        };
    }
    match type_ {
        ScSpecTypeDef::Bool => value
            .parse::<bool>()
            .map(|v| v.into_val(env))
            .map_err(|_| CliError(format!("invalid bool value {value:?}; use true or false"))),
        ScSpecTypeDef::U32 => Ok(parse!(u32)),
        ScSpecTypeDef::I32 => Ok(parse!(i32)),
        ScSpecTypeDef::U64 => Ok(parse!(u64)),
        ScSpecTypeDef::I64 => Ok(parse!(i64)),
        ScSpecTypeDef::U128 => Ok(parse!(u128)),
        ScSpecTypeDef::I128 => Ok(parse!(i128)),
        ScSpecTypeDef::Symbol => Symbol::try_from_val(env, &value)
            .map(|symbol| symbol.into_val(env))
            .map_err(|err| CliError(format!("invalid Symbol value {value:?}: {err:?}"))),
        ScSpecTypeDef::String => Ok(soroban_sdk::String::from_str(env, value).into_val(env)),
        ScSpecTypeDef::Bytes => {
            let hex = value
                .strip_prefix("0x")
                .ok_or_else(|| CliError("Bytes --arg values must start with 0x".to_string()))?;
            let bytes = decode_hex(hex)?;
            // An empty override (0x with no digits) must produce the same
            // empty `Bytes` that `default_val` and `vec_element_val` use
            // (`Bytes::new`), not `Bytes::from_slice(&[])`, which serializes
            // to invalid XDR for the zero-length case.
            if bytes.is_empty() {
                Ok(soroban_sdk::Bytes::new(env).into_val(env))
            } else {
                Ok(soroban_sdk::Bytes::from_slice(env, &bytes).into_val(env))
            }
        }
        ScSpecTypeDef::Address => {
            let strkey = value
                .parse::<stellar_strkey::Strkey>()
                .map_err(|err| CliError(format!("invalid Stellar address {value:?}: {err}")))?;
            if !matches!(
                strkey,
                stellar_strkey::Strkey::PublicKeyEd25519(_) | stellar_strkey::Strkey::Contract(_)
            ) {
                return Err(CliError(format!(
                    "{value:?} is a valid strkey but not an account or contract address"
                )));
            }
            Ok(Address::from_str(env, value).into_val(env))
        }
        other => Err(CliError(format!(
            "--arg does not support explicit values for {other:?}"
        ))),
    }
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, CliError> {
    if !hex.len().is_multiple_of(2) {
        return Err(CliError(
            "Bytes --arg hex must contain pairs of digits".to_string(),
        ));
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|_| CliError(format!("invalid hexadecimal Bytes value {hex:?}")))
        })
        .collect()
}

fn invoke_succeeds(env: &Env, contract_id: &Address, func: &Symbol, args: SVec<Val>) -> bool {
    // try_invoke_contract is the SDK's own no-panic call path, used here
    // so a probe that succeeds doesn't need catch_unwind at all â only a
    // *failing* real-WASM call risks the process abort documented above,
    // and that only ever happens in the isolated child process.
    matches!(
        env.try_invoke_contract::<Val, soroban_sdk::Error>(contract_id, func, args),
        Ok(Ok(_))
    )
}

fn ramp_val(env: &Env, type_: &ScSpecTypeDef, ramp_value: u32) -> Result<Val, CliError> {
    match type_ {
        ScSpecTypeDef::U32 => Ok(ramp_value.into_val(env)),
        // Signed integer types are supported: the internal ramp counter is a
        // u32 (0..=SEARCH_UPPER_BOUND), cast to the target signed type. For
        // values ≤ i32::MAX / i64::MAX / i128::MAX this is a lossless widening
        // cast that produces the expected positive ramp sequence. Values above
        // i32::MAX would wrap negative on cast to i32; in practice the ramp
        // ceiling for a signed-integer parameter is found well below that bound.
        ScSpecTypeDef::I32 => Ok((ramp_value as i32).into_val(env)),
        ScSpecTypeDef::U64 => Ok((ramp_value as u64).into_val(env)),
        ScSpecTypeDef::I64 => Ok((ramp_value as i64).into_val(env)),
        ScSpecTypeDef::U128 => Ok((ramp_value as u128).into_val(env)),
        ScSpecTypeDef::I128 => Ok((ramp_value as i128).into_val(env)),
        ScSpecTypeDef::Bytes => {
            let bytes = vec![0u8; ramp_value as usize];
            Ok(soroban_sdk::Bytes::from_slice(env, &bytes).into_val(env))
        }
        // Any Vec<T> is ramped as its element *count*, filled with a fixed
        // per-element value from `vec_element_val` â the same "maximum
        // recipients" shape as the original Vec<Address>-only support, now
        // generalized to any element type this command already knows how
        // to default. The element value itself never varies across the
        // ramp; only the count does.
        ScSpecTypeDef::Vec(inner) => {
            let element = vec_element_val(env, &inner.element_type)?;
            let mut items = SVec::new(env);
            for _ in 0..ramp_value {
                items.push_back(element);
            }
            Ok(items.into_val(env))
        }
        other => Err(CliError(format!(
            "the ramp parameter's type ({other:?}) isn't supported yet; supported ramp types \
            are u32/i32/u64/i64/u128/i128, Bytes, and Vec<T> (for the element types listed in \
             the Vec<T> element-type error, if T itself isn't supported)"
        ))),
    }
}

/// A fixed filler value for one element of a ramped `Vec<T>`. Deliberately
/// narrower than [`default_val`]: it has no `admin`/`input` context (an
/// element has no parameter name to apply the `token`-heuristic to, and
/// nothing meaningfully "ramps" as a nested Vec-of-Vec's inner count), so
/// unsupported element types are reported explicitly rather than silently
/// reusing a heuristic that wouldn't make sense at this level.
fn vec_element_val(env: &Env, type_: &ScSpecTypeDef) -> Result<Val, CliError> {
    match type_ {
        ScSpecTypeDef::Address => Ok(Address::generate(env).into_val(env)),
        ScSpecTypeDef::Bool => Ok(false.into_val(env)),
        ScSpecTypeDef::U32 => Ok(1u32.into_val(env)),
        ScSpecTypeDef::I32 => Ok(1i32.into_val(env)),
        ScSpecTypeDef::U64 => Ok(1u64.into_val(env)),
        ScSpecTypeDef::I64 => Ok(1i64.into_val(env)),
        ScSpecTypeDef::U128 => Ok(1u128.into_val(env)),
        ScSpecTypeDef::I128 => Ok(1i128.into_val(env)),
        ScSpecTypeDef::Symbol => Ok(Symbol::new(env, "x").into_val(env)),
        ScSpecTypeDef::String => Ok(soroban_sdk::String::from_str(env, "").into_val(env)),
        ScSpecTypeDef::Bytes => Ok(soroban_sdk::Bytes::new(env).into_val(env)),
        other => Err(CliError(format!(
            "Vec<{other:?}> isn't supported as a ramp parameter yet; supported Vec<T> element \
             types are Address/Bool/u32/i32/u64/i64/u128/i128/Symbol/String/Bytes"
        ))),
    }
}

fn default_val(
    env: &Env,
    admin: &Address,
    type_: &ScSpecTypeDef,
    input: &ScSpecFunctionInputV0,
) -> Result<Val, CliError> {
    match type_ {
        // A parameter literally named `token` is, by overwhelming Soroban
        // convention, a token contract address â and a batch/payout-style
        // function will call transfer() on it and needs `admin` to hold a
        // real balance, not just a syntactically valid address. Deploying
        // a funded Stellar Asset Contract for this one case is a
        // name-based heuristic, not a semantic guarantee, but it's what
        // makes `limits` usable against a real payout contract at all
        // (verified against sororail-contracts' batch_payout, whose
        // execute_equal(funder, token, recipients, amount_each) is
        // exactly this shape).
        ScSpecTypeDef::Address if input.name.to_utf8_string_lossy() == "token" => {
            let issuer = Address::generate(env);
            let sac = env.register_stellar_asset_contract_v2(issuer.clone());
            let token_address = sac.address();
            let amount = i128::MAX / 2;
            StellarAssetClient::new(env, &token_address)
                .mock_auths(&[MockAuth {
                    address: &issuer,
                    invoke: &MockAuthInvoke {
                        contract: &token_address,
                        fn_name: "mint",
                        args: (admin.clone(), amount).into_val(env),
                        sub_invokes: &[],
                    },
                }])
                .mint(admin, &amount);
            Ok(token_address.into_val(env))
        }
        ScSpecTypeDef::Address => Ok(admin.into_val(env)),
        ScSpecTypeDef::Bool => Ok(false.into_val(env)),
        // 1, not 0: verified against sororail-contracts' batch_payout,
        // whose execute_equal(..., amount_each: i128) requires a positive
        // amount and rejects 0 with its own InvalidAmount error before
        // any resource-limit-relevant work happens. There's no default
        // that's safe for every contract's validation rules, but 1 is
        // the smaller mistake: far more contracts reject a non-positive
        // amount than reject exactly 1.
        ScSpecTypeDef::U32 => Ok(1u32.into_val(env)),
        ScSpecTypeDef::I32 => Ok(1i32.into_val(env)),
        ScSpecTypeDef::U64 => Ok(1u64.into_val(env)),
        ScSpecTypeDef::I64 => Ok(1i64.into_val(env)),
        ScSpecTypeDef::U128 => Ok(1u128.into_val(env)),
        ScSpecTypeDef::I128 => Ok(1i128.into_val(env)),
        ScSpecTypeDef::Symbol => Ok(Symbol::new(env, "x").into_val(env)),
        ScSpecTypeDef::String => Ok(soroban_sdk::String::from_str(env, "").into_val(env)),
        ScSpecTypeDef::Bytes => Ok(soroban_sdk::Bytes::new(env).into_val(env)),
        ScSpecTypeDef::Vec(_) => Ok(SVec::<Val>::new(env).into_val(env)),
        other => Err(CliError(format!(
            "parameter {:?} has type {other:?}, which this command doesn't know how to \
             default yet; only the --ramp parameter needs a type this command understands \
             today, every other parameter needs a supported default type",
            input.name.to_utf8_string_lossy()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soroban_sdk::xdr::ScSpecTypeVec;
    use std::collections::BTreeMap;
    use std::str::FromStr;

    /// `LimitsArgs` derives `clap::Args` (it is flattened into the top-level
    /// CLI), which has no `try_parse_from`. Parse through a one-field wrapper
    /// so the flag tests exercise the real clap definitions.
    #[derive(clap::Parser)]
    struct LimitsCli {
        #[command(flatten)]
        args: LimitsArgs,
    }

    impl LimitsArgs {
        fn try_parse_from<I, T>(argv: I) -> Result<Self, clap::Error>
        where
            I: IntoIterator<Item = T>,
            T: Into<std::ffi::OsString> + Clone,
        {
            <LimitsCli as clap::Parser>::try_parse_from(argv).map(|cli| cli.args)
        }
    }

    // ---- #239: Vec<T> ramp support ----

    #[test]
    fn ramp_val_vec_address_produces_the_requested_count() {
        let env = Env::default();
        let type_ = ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
            element_type: Box::new(ScSpecTypeDef::Address),
        }));
        let val = ramp_val(&env, &type_, 5).unwrap();
        let vec: SVec<Address> = val.into_val(&env);
        assert_eq!(vec.len(), 5);
    }

    #[test]
    fn ramp_val_vec_u32_produces_the_requested_count() {
        let env = Env::default();
        let type_ = ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
            element_type: Box::new(ScSpecTypeDef::U32),
        }));
        let val = ramp_val(&env, &type_, 7).unwrap();
        let vec: SVec<u32> = val.into_val(&env);
        assert_eq!(vec.len(), 7);
        assert_eq!(vec.get(0), Some(1u32));
    }

    #[test]
    fn ramp_val_vec_of_zero_is_an_empty_vec() {
        let env = Env::default();
        let type_ = ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
            element_type: Box::new(ScSpecTypeDef::Symbol),
        }));
        let val = ramp_val(&env, &type_, 0).unwrap();
        let vec: SVec<Symbol> = val.into_val(&env);
        assert!(vec.is_empty());
    }

    #[test]
    fn ramp_val_rejects_vec_of_unsupported_element_type() {
        let env = Env::default();
        let type_ = ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
            element_type: Box::new(ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
                element_type: Box::new(ScSpecTypeDef::U32),
            }))),
        }));
        let err = ramp_val(&env, &type_, 3).unwrap_err();
        assert!(err.0.contains("isn't supported"), "{}", err.0);
    }

    #[test]
    fn ramp_val_rejects_an_unsupported_scalar_type() {
        let env = Env::default();
        let err = ramp_val(&env, &ScSpecTypeDef::Void, 1).unwrap_err();
        assert!(err.0.contains("isn't supported yet"), "{}", err.0);
    }

    #[test]
    fn ramp_val_bytes_uses_the_requested_length() {
        let env = Env::default();
        let val = ramp_val(&env, &ScSpecTypeDef::Bytes, 4).unwrap();
        let bytes: soroban_sdk::Bytes = val.into_val(&env);
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes.get(0), Some(0));
    }

    #[test]
    fn explicit_values_parse_typed_scalars_and_bytes() {
        let env = Env::default();
        let value: u32 = explicit_val(&env, &ScSpecTypeDef::U32, "42")
            .unwrap()
            .into_val(&env);
        assert_eq!(value, 42);
        let bytes: soroban_sdk::Bytes = explicit_val(&env, &ScSpecTypeDef::Bytes, "0x00a1")
            .unwrap()
            .into_val(&env);
        assert_eq!(bytes.len(), 2);
        assert_eq!(
            decode_hex("abc").unwrap_err().0,
            "Bytes --arg hex must contain pairs of digits"
        );
    }

    #[test]
    fn ramp_steps_stop_at_the_cap_without_overflow() {
        assert_eq!(next_ramp_value(1, SEARCH_UPPER_BOUND), Some(2));
        assert_eq!(
            next_ramp_value(SEARCH_UPPER_BOUND / 2, SEARCH_UPPER_BOUND),
            Some(SEARCH_UPPER_BOUND)
        );
        assert_eq!(
            next_ramp_value(SEARCH_UPPER_BOUND, SEARCH_UPPER_BOUND),
            None
        );
        assert_eq!(next_ramp_value(8, 13), Some(13));
        assert_eq!(next_ramp_value(u32::MAX / 2 + 1, u32::MAX), Some(u32::MAX));
        assert_eq!(next_ramp_value(u32::MAX, u32::MAX), None);
        assert_eq!(next_ramp_value(u32::MAX, SEARCH_UPPER_BOUND), None);
    }

    #[test]
    fn refinement_rejects_changed_endpoints_before_probing_interior() {
        for (lower_succeeds, upper_succeeds, expected, calls) in [
            (false, false, "lower bound 8 no longer succeeds", vec![8]),
            (true, true, "upper bound 32 now succeeds", vec![8, 32]),
            (false, true, "lower bound 8 no longer succeeds", vec![8]),
        ] {
            let mut probed = Vec::new();
            let err = refine_upper_bound(8, 32, &mut |value| {
                probed.push(value);
                Ok(if value == 8 {
                    lower_succeeds
                } else {
                    upper_succeeds
                })
            })
            .unwrap_err();
            assert!(err.0.contains("non-monotonic probe results"), "{}", err.0);
            assert!(err.0.contains(expected), "{}", err.0);
            assert_eq!(probed, calls);
        }
    }

    #[test]
    fn refinement_validates_adjacent_bounds_and_handles_u32_max() {
        let (low, high, audit) = refine_upper_bound_audited(u32::MAX - 1, u32::MAX, &mut |value| {
            Ok(if value < u32::MAX {
                ProbeOutcome::Succeeded
            } else {
                ProbeOutcome::Failed(ProbeFailureCategory::ExitFailure)
            })
        })
        .unwrap();
        assert_eq!((low, high), (u32::MAX - 1, u32::MAX));
        assert!(audit.steps.is_empty());
        let result = refine_upper_bound(1, u32::MAX, &mut |v| Ok(v < u32::MAX)).unwrap();
        assert_eq!(result, (u32::MAX - 1, u32::MAX));
    }

    #[test]
    fn refinement_preserves_probe_errors() {
        let err = refine_upper_bound(1, 4, &mut |_| Err(CliError("cannot spawn probe".into())))
            .unwrap_err();
        assert_eq!(err.0, "cannot spawn probe");
    }

    #[test]
    fn binary_search_refines_to_adjacent_success_and_failure() {
        let mut probed = Vec::new();
        let (success, failure) = refine_upper_bound(8, 32, &mut |value| {
            probed.push(value);
            Ok(value <= 19)
        })
        .unwrap();
        assert_eq!((success, failure), (19, 20));
        assert_eq!(&probed[..2], &[8, 32]);
        assert!(probed.len() <= 7);
    }

    // ---- #245: baseline comparison ----

    fn sample_baseline() -> Baseline {
        Baseline {
            contract: Some("vault.wasm".to_string()),
            function: "batch_payout".to_string(),
            ramp: "recipients".to_string(),
            best: 100,
            instructions: 1_000_000,
            memory_bytes: 50_000,
            measurement_limitations: Some(MEASUREMENT_LIMITATIONS.to_string()),
        }
    }

    fn sample_summary() -> LimitsSummary {
        LimitsSummary {
            contract: "vault.wasm".to_string(),
            function: Some("batch_payout".to_string()),
            ramp: Some("recipients".to_string()),
            best: 100,
            instructions: 1_000_000,
            memory_bytes: 50_000,
            first_failed: Some(101),
            first_failure_category: Some(ProbeFailureCategory::ExitFailure),
            first_failed_probe_stderr: None,
            last_failed_probe: None,
            probe_failure_counts: BTreeMap::from([(ProbeFailureCategory::ExitFailure, 1)]),
            hit_upper_bound: false,
            probe_timeout_secs: 30,
            audit: None,
            measurement_limitations: MEASUREMENT_LIMITATIONS.to_string(),
            reproduction: Some(
                "invoke `batch_payout` with `recipients` = 100 in a test environment; to re-verify the boundary via CLI run: `soroban-testkit limits --contract vault.wasm --fn batch_payout --ramp recipients`".to_string(),
            ),
            stopped_by: None,
            probe_samples: Vec::new(),
            co_ramp: Vec::new(),
            compiler_meta: None,
            wasm_target_dir: None,
            profile: None,
        }
    }

    /// A summary whose contract field went through [`sanitize_report_path`],
    /// mirroring what `run` stores before exporting.
    fn summary_for_contract(contract: &Path) -> LimitsSummary {
        LimitsSummary {
            contract: sanitize_report_path(contract),
            ..sample_summary()
        }
    }

    #[test]
    fn baseline_round_trips_through_its_file_format() {
        let baseline = sample_baseline();
        let parsed = Baseline::parse(&baseline.to_file_contents()).unwrap();
        assert_eq!(parsed.contract, baseline.contract);
        assert_eq!(parsed.function, baseline.function);
        assert_eq!(parsed.ramp, baseline.ramp);
        assert_eq!(parsed.best, baseline.best);
        assert_eq!(parsed.instructions, baseline.instructions);
        assert_eq!(parsed.memory_bytes, baseline.memory_bytes);
        assert_eq!(
            parsed.measurement_limitations,
            baseline.measurement_limitations
        );
    }

    #[test]
    fn baseline_parse_rejects_a_missing_field() {
        let err = Baseline::parse("function=f\nramp=r\nbest=1\n").unwrap_err();
        assert!(err.0.contains("missing `instructions`"), "{}", err.0);
    }

    #[test]
    fn baseline_parse_rejects_an_unknown_key() {
        let err = Baseline::parse("function=f\nramp=r\nbest=1\nbogus=2\n").unwrap_err();
        assert!(err.0.contains("unknown key"), "{}", err.0);
    }

    #[test]
    fn baseline_parse_rejects_a_malformed_line() {
        let err = Baseline::parse("not-a-key-value-line\n").unwrap_err();
        assert!(err.0.contains("malformed baseline file"), "{}", err.0);
    }

    #[test]
    fn compare_to_baseline_passes_when_nothing_regressed() {
        let dir = std::env::temp_dir().join(format!("stk-baseline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pass.baseline");
        std::fs::write(&path, sample_baseline().to_file_contents()).unwrap();

        let result = compare_to_baseline(
            &path,
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
            5.0,
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn compare_to_baseline_fails_when_the_ramp_ceiling_drops() {
        let dir = std::env::temp_dir().join(format!("stk-baseline-{}-2", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ceiling-drop.baseline");
        std::fs::write(&path, sample_baseline().to_file_contents()).unwrap();

        let err = compare_to_baseline(
            &path,
            "recipients",
            "batch_payout",
            90,
            1_000_000,
            50_000,
            5.0,
        )
        .unwrap_err();
        assert!(
            err.0.contains("ceiling dropped from 100 to 90"),
            "{}",
            err.0
        );
    }

    #[test]
    fn compare_to_baseline_fails_when_instructions_grow_past_tolerance() {
        let dir = std::env::temp_dir().join(format!("stk-baseline-{}-3", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("instr-grow.baseline");
        std::fs::write(&path, sample_baseline().to_file_contents()).unwrap();

        // +10% instructions, tolerance is 5%.
        let err = compare_to_baseline(
            &path,
            "recipients",
            "batch_payout",
            100,
            1_100_000,
            50_000,
            5.0,
        )
        .unwrap_err();
        assert!(err.0.contains("instructions grew"), "{}", err.0);
    }

    #[test]
    fn compare_to_baseline_allows_growth_within_tolerance() {
        let dir = std::env::temp_dir().join(format!("stk-baseline-{}-4", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("within-tolerance.baseline");
        std::fs::write(&path, sample_baseline().to_file_contents()).unwrap();

        // +2% instructions, tolerance is 5%.
        let result = compare_to_baseline(
            &path,
            "recipients",
            "batch_payout",
            100,
            1_020_000,
            50_000,
            5.0,
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn compare_to_baseline_rejects_a_mismatched_function_or_ramp() {
        let dir = std::env::temp_dir().join(format!("stk-baseline-{}-5", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mismatch.baseline");
        std::fs::write(&path, sample_baseline().to_file_contents()).unwrap();

        let err = compare_to_baseline(&path, "amount", "batch_payout", 100, 1_000_000, 50_000, 5.0)
            .unwrap_err();
        assert!(err.0.contains("was recorded for"), "{}", err.0);
    }

    #[test]
    fn compare_to_baseline_reports_a_missing_file_actionably() {
        let missing = std::env::temp_dir().join("this-baseline-does-not-exist.baseline");
        let err = compare_to_baseline(
            &missing,
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
            5.0,
        )
        .unwrap_err();
        assert!(err.0.contains("failed to read baseline"), "{}", err.0);
    }

    // ---- #243: per-probe timeout ----

    #[test]
    fn wait_with_timeout_returns_the_exit_status_for_a_fast_process() {
        let mut child = Command::new("true").spawn().unwrap();
        let status = wait_with_timeout(&mut child, Duration::from_secs(5))
            .unwrap()
            .expect("should not have timed out");
        assert!(status.success());
    }

    #[test]
    fn baseline_tolerance_must_be_between_zero_and_one_hundred() {
        for pct in [-0.1, 100.1, f64::NAN, f64::INFINITY] {
            let err = validate_percentage("--baseline-tolerance-pct", pct).unwrap_err();
            assert!(err.0.contains("between 0 and 100"), "{}", err.0);
        }
    }

    #[test]
    fn wait_with_timeout_kills_and_returns_none_for_a_slow_process() {
        let mut child = Command::new("sleep").arg("5").spawn().unwrap();
        let start = Instant::now();
        let result = wait_with_timeout(&mut child, Duration::from_millis(200)).unwrap();
        assert!(result.is_none());
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "should have killed the child near the timeout, not waited for it to finish"
        );
    }

    #[test]
    fn shell_quote_protects_paths_with_spaces_and_single_quotes() {
        assert_eq!(shell_quote("vault.wasm"), "vault.wasm");
        assert_eq!(shell_quote("my dir/vault.wasm"), "'my dir/vault.wasm'");
        assert_eq!(shell_quote("can't run.wasm"), "'can'\\''t run.wasm'");
        assert_eq!(shell_quote("$(touch bad)"), "'$(touch bad)'");
        let guide = build_reproduction_guidance(
            "my dir/can't run.wasm",
            "batch_payout",
            "recipients",
            3,
            &["memo=hello world".into()],
        );
        assert!(guide.contains("--contract 'my dir/can'\\''t run.wasm'"));
        assert!(guide.contains("--arg 'memo=hello world'"));
    }

    #[test]
    fn completed_probe_retains_actual_nonzero_exit_status() {
        let mut child = Command::new("sh").args(["-c", "exit 17"]).spawn().unwrap();
        let status = wait_with_timeout(&mut child, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(status.code(), Some(17));
        assert_eq!(child.try_wait().unwrap().unwrap().code(), Some(17));
    }

    #[test]
    fn timed_out_probe_is_killed_and_reaped() {
        let mut child = Command::new("sleep").arg("10").spawn().unwrap();
        assert!(wait_with_timeout(&mut child, Duration::from_millis(40))
            .unwrap()
            .is_none());
        assert!(child.try_wait().unwrap().is_some(), "child must be reaped");
    }

    #[test]
    fn abandoned_probe_guard_prevents_late_artifacts() {
        let root = std::env::temp_dir().join(format!("stk-interrupted-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("late-artifact");
        let child = Command::new("sh")
            .args(["-c", "sleep 1; touch \"$1\"", "_"])
            .arg(&marker)
            .current_dir(&root)
            .spawn()
            .unwrap();
        {
            // Simulates an early return/error: the guard kills the child.
            let _probe = ProbeChild { child };
        }
        std::thread::sleep(Duration::from_millis(1100));
        assert!(!marker.exists(), "an abandoned probe wrote after Drop");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        std::fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn oversized_timeout_is_actionable_and_guard_still_reaps() {
        let child = Command::new("sleep").arg("10").spawn().unwrap();
        let mut guard = ProbeChild { child };
        let err = wait_with_timeout(&mut guard.child, Duration::MAX).unwrap_err();
        assert!(err.0.contains("too large for the host clock"), "{}", err.0);
        drop(guard);
    }

    // ---- #244: export results ----

    #[test]
    fn export_results_writes_a_file_with_the_correct_format() {
        let dir = std::env::temp_dir().join(format!("stk-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export.results");

        export_results(&path, &sample_summary()).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("contract=vault.wasm"));
        assert!(contents.contains("function=batch_payout"));
        assert!(contents.contains("ramp=recipients"));
        assert!(contents.contains("best=100"));
        assert!(contents.contains("instructions=1000000"));
        assert!(contents.contains("memory_bytes=50000"));
        assert!(contents.contains(&format!(
            "measurement_limitations={MEASUREMENT_LIMITATIONS}"
        )));
    }

    #[test]
    fn export_results_can_be_parsed_as_a_baseline() {
        let dir = std::env::temp_dir().join(format!("stk-export-{}-2", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export.results");

        export_results(&path, &sample_summary()).unwrap();

        let baseline = Baseline::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(baseline.contract, Some("vault.wasm".to_string()));
        assert_eq!(baseline.function, "batch_payout");
        assert_eq!(baseline.ramp, "recipients");
        assert_eq!(baseline.best, 100);
        assert_eq!(baseline.instructions, 1_000_000);
        assert_eq!(baseline.memory_bytes, 50_000);
    }

    // ---- #233: paths with spaces ----

    #[test]
    fn write_baseline_handles_paths_with_spaces() {
        let dir = std::env::temp_dir().join(format!("stk baseline test {}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("my baseline.txt");

        write_baseline(
            &path,
            Path::new("vault.wasm"),
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
        )
        .unwrap();

        let baseline = Baseline::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(baseline.contract, Some("vault.wasm".to_string()));
        assert_eq!(baseline.function, "batch_payout");
        assert_eq!(baseline.ramp, "recipients");
        assert_eq!(baseline.best, 100);
    }

    #[test]
    fn export_results_handles_paths_with_spaces() {
        let dir = std::env::temp_dir().join(format!("stk export test {}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("my export results.txt");

        export_results(&path, &summary_for_contract(Path::new("vault.wasm"))).unwrap();

        let baseline = Baseline::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(baseline.contract, Some("vault.wasm".to_string()));
        assert_eq!(baseline.function, "batch_payout");
        assert_eq!(baseline.ramp, "recipients");
        assert_eq!(baseline.best, 100);
    }

    #[test]
    fn compare_to_baseline_handles_paths_with_spaces() {
        let dir = std::env::temp_dir().join(format!("stk compare test {}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("my baseline file.txt");

        std::fs::write(&path, sample_baseline().to_file_contents()).unwrap();

        let result = compare_to_baseline(
            &path,
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
            5.0,
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // ---- #227: handle missing WASM exports ----

    #[test]
    fn load_rejects_missing_function_with_available_list() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("stk-load-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let wasm_path = dir.join("test.wasm");
        let mut file = std::fs::File::create(&wasm_path).unwrap();
        file.write_all(b"mock").unwrap();

        let err_msg = match load(&wasm_path, "nonexistent", "count") {
            Err(e) => e.0,
            Ok(_) => panic!("expected error"),
        };
        assert!(
            err_msg.contains("failed to read contract spec"),
            "expected spec error, got: {}",
            err_msg
        );
    }

    #[test]
    fn load_provides_helpful_param_error() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("stk-param-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let wasm_path = dir.join("test.wasm");
        let mut file = std::fs::File::create(&wasm_path).unwrap();
        file.write_all(b"mock").unwrap();

        let err_msg = match load(&wasm_path, "test_func", "wrong_param") {
            Err(e) => e.0,
            Ok(_) => panic!("expected error"),
        };
        assert!(
            err_msg.contains("failed to read contract spec"),
            "expected spec error early, got: {}",
            err_msg
        );
    }

    // ---- #228: distinguish resource-limit failures from validation ----

    #[test]
    fn run_distinguishes_validation_failures_from_limits() {
        let err_result = LimitsArgs {
            contract: std::path::PathBuf::from("nonexistent.wasm"),
            function: Some("test".to_string()),
            ramp: Some("count".to_string()),
            probe_timeout: 30,
            timeout: None,
            progress: false,
            jobs: 1,
            baseline: None,
            baseline_tolerance_pct: 5.0,
            save_baseline: None,
            export: None,
            preset: None,
            budget: None,
            arg_values: Vec::new(),
            co_ramp: Vec::new(),
            dry_run: false,
            export_samples: None,
            max_probes: 0,
            wasm_target_dir: None,
            profile: None,
            search_strategy: SearchStrategy::Doubling,
            step: 1,
            record_probe_duration: false,
        };

        let result = run(err_result);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().0;
        assert!(
            err_msg.contains("failed to read"),
            "expected file error, got: {}",
            err_msg
        );
    }

    // ---- #192: opt-in bounded parallel probing ----

    fn args_with_jobs(jobs: usize) -> LimitsArgs {
        LimitsArgs {
            contract: std::path::PathBuf::from("nonexistent.wasm"),
            function: Some("test".to_string()),
            ramp: Some("count".to_string()),
            probe_timeout: 30,
            timeout: None,
            progress: false,
            jobs,
            baseline: None,
            baseline_tolerance_pct: 5.0,
            save_baseline: None,
            export: None,
            preset: None,
            budget: None,
            arg_values: Vec::new(),
            co_ramp: Vec::new(),
            dry_run: false,
            export_samples: None,
            max_probes: 0,
            wasm_target_dir: None,
            profile: None,
            search_strategy: SearchStrategy::Doubling,
            step: 1,
            record_probe_duration: false,
        }
    }

    #[test]
    fn jobs_outside_one_to_eight_is_rejected_before_loading_the_contract() {
        for jobs in [0, MAX_PROBE_JOBS + 1] {
            let err = run(args_with_jobs(jobs)).unwrap_err().0;
            assert_eq!(err, format!("--jobs must be between 1 and 8 (got {jobs})"));
        }
    }

    #[test]
    fn ladder_batch_takes_consecutive_ladder_values_up_to_jobs() {
        assert_eq!(ladder_batch(1, SEARCH_UPPER_BOUND, 1), vec![1]);
        assert_eq!(ladder_batch(1, SEARCH_UPPER_BOUND, 3), vec![1, 2, 4]);
        assert_eq!(ladder_batch(4, SEARCH_UPPER_BOUND, 2), vec![4, 8]);
    }

    #[test]
    fn ladder_batch_stops_at_the_upper_bound() {
        assert_eq!(ladder_batch(4, 8, 8), vec![4, 8]);
        assert_eq!(ladder_batch(8, 8, 4), vec![8]);
    }

    #[test]
    fn ladder_batch_is_never_empty_even_for_zero_jobs() {
        assert_eq!(ladder_batch(2, SEARCH_UPPER_BOUND, 0), vec![2]);
    }

    // ---- #493: numeric ramp boundary refinement audit ----

    #[test]
    fn refine_upper_bound_audited_records_all_steps_and_boundaries() {
        let mut probed = Vec::new();
        let (success, failure, audit) = refine_upper_bound_audited(10, 30, &mut |value| {
            probed.push(value);
            Ok(if value <= 23 {
                ProbeOutcome::Succeeded
            } else {
                ProbeOutcome::Failed(ProbeFailureCategory::ExitFailure)
            })
        })
        .unwrap();

        assert_eq!((success, failure), (23, 24));
        assert_eq!(audit.initial_low, 10);
        assert_eq!(audit.initial_high, 30);
        assert_eq!(audit.final_success, 23);
        assert_eq!(audit.final_failure, 24);
        assert_eq!(&probed[..2], &[10, 30]);
        assert_eq!(audit.steps.len(), probed.len() - 2);

        for (i, step) in audit.steps.iter().enumerate() {
            assert_eq!(step.step, i + 1);
            assert_eq!(step.probe_value, probed[i + 2]);
            assert_eq!(step.succeeded, step.probe_value <= 23);
            // Every failing step carries the standardized category; every
            // succeeding step carries none.
            assert_eq!(
                step.failure_category,
                (!step.succeeded).then_some(ProbeFailureCategory::ExitFailure)
            );
        }
    }

    // ---- #444: contract failures must not be treated as budget ceilings ----
    #[test]
    fn probe_exit_classifier_distinguishes_contract_errors_from_unknown_exits() {
        assert_eq!(
            classify_probe_exit_stderr(
                "error: contract-failure: invocation returned Error(Contract, #1)\n"
            ),
            ProbeFailureCategory::ContractFailure
        );
        assert_eq!(
            classify_probe_exit_stderr("error: probe failed\n"),
            ProbeFailureCategory::ExitFailure
        );
        assert_eq!(
            classify_probe_exit_stderr(""),
            ProbeFailureCategory::ExitFailure
        );
        assert_eq!(
            serde_json::to_value(ProbeFailureCategory::ContractFailure).unwrap(),
            serde_json::json!("contract-failure")
        );
    }

    // ---- #492: standardized probe failure categories ----

    #[test]
    fn probe_failure_categories_have_stable_labels_and_guidance() {
        let expected = [
            (ProbeFailureCategory::ExitFailure, "exit-failure"),
            (ProbeFailureCategory::TimedOut, "timed-out"),
            (ProbeFailureCategory::Abort, "abort"),
            (ProbeFailureCategory::SpawnFailure, "spawn-failure"),
        ];
        for (category, label) in expected {
            assert_eq!(category.label(), label);
            assert_eq!(category.to_string(), label);
            assert!(
                !category.guidance().is_empty(),
                "category {label} must carry actionable guidance"
            );
        }
    }

    #[test]
    fn probe_failure_categories_serialize_as_stable_kebab_case_strings() {
        let json = serde_json::to_value(ProbeFailureCategory::TimedOut).unwrap();
        assert_eq!(json, serde_json::json!("timed-out"));
        let parsed: ProbeFailureCategory = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, ProbeFailureCategory::TimedOut);
    }

    #[test]
    fn probe_outcome_reports_success_only_for_succeeded() {
        assert!(ProbeOutcome::Succeeded.succeeded());
        assert!(!ProbeOutcome::Failed(ProbeFailureCategory::Abort).succeeded());
    }

    #[test]
    fn refine_upper_bound_rejects_inverted_or_equal_bounds() {
        let err_inverted = refine_upper_bound(30, 20, &mut |_| Ok(true)).unwrap_err();
        assert!(
            err_inverted
                .0
                .contains("requires lower bound (30) < upper bound (20)"),
            "{}",
            err_inverted.0
        );

        let err_equal = refine_upper_bound(20, 20, &mut |_| Ok(true)).unwrap_err();
        assert!(
            err_equal
                .0
                .contains("requires lower bound (20) < upper bound (20)"),
            "{}",
            err_equal.0
        );
    }

    // ---- #494: sanitize temporary paths from machine-readable reports ----

    #[test]
    fn sanitize_report_path_strips_temp_directory_prefixes() {
        let temp = std::env::temp_dir().join("some_run_1234/contract.wasm");
        let sanitized = sanitize_report_path(&temp);
        assert_eq!(sanitized, "contract.wasm");

        let tmp_path = Path::new("/tmp/some_dir/mock.wasm");
        assert_eq!(sanitize_report_path(tmp_path), "mock.wasm");

        let rel_path = Path::new("relative/path/contract.wasm");
        assert_eq!(
            sanitize_report_path(rel_path),
            "relative/path/contract.wasm"
        );
    }

    #[test]
    fn write_baseline_sanitizes_temporary_contract_path() {
        let dir = std::env::temp_dir().join(format!("stk-sanitize-base-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clean.baseline");
        let temp_contract = dir.join("ephemeral/vault.wasm");

        write_baseline(
            &path,
            &temp_contract,
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
        )
        .unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(
            contents.contains("contract=vault.wasm"),
            "contents: {contents}"
        );
        assert!(
            !contents.contains("stk-sanitize-base"),
            "leaked temp dir: {contents}"
        );
    }

    #[test]
    fn export_results_json_sanitizes_temporary_contract_path() {
        let dir = std::env::temp_dir().join(format!("stk-sanitize-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let temp_contract = dir.join("run-pid-999/vault.wasm");

        export_results(&path, &summary_for_contract(&temp_contract)).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["contract"], "vault.wasm");
        assert!(!contents.contains("run-pid-999"));
    }

    // ---- #495: regression coverage for each supported ramp type ----

    #[test]
    fn ramp_val_supports_all_scalar_numeric_types() {
        let env = Env::default();

        // U32
        let val = ramp_val(&env, &ScSpecTypeDef::U32, 42).unwrap();
        let num: u32 = val.into_val(&env);
        assert_eq!(num, 42);

        // I32
        let val = ramp_val(&env, &ScSpecTypeDef::I32, 42).unwrap();
        let num: i32 = val.into_val(&env);
        assert_eq!(num, 42);

        // U64
        let val = ramp_val(&env, &ScSpecTypeDef::U64, 42).unwrap();
        let num: u64 = val.into_val(&env);
        assert_eq!(num, 42);

        // I64
        let val = ramp_val(&env, &ScSpecTypeDef::I64, 42).unwrap();
        let num: i64 = val.into_val(&env);
        assert_eq!(num, 42);

        // U128
        let val = ramp_val(&env, &ScSpecTypeDef::U128, 42).unwrap();
        let num: u128 = val.into_val(&env);
        assert_eq!(num, 42);

        // I128
        let val = ramp_val(&env, &ScSpecTypeDef::I128, 42).unwrap();
        let num: i128 = val.into_val(&env);
        assert_eq!(num, 42);
    }

    #[test]
    fn ramp_val_supports_all_vec_element_types() {
        let env = Env::default();

        let element_types = [
            ScSpecTypeDef::Address,
            ScSpecTypeDef::Bool,
            ScSpecTypeDef::U32,
            ScSpecTypeDef::I32,
            ScSpecTypeDef::U64,
            ScSpecTypeDef::I64,
            ScSpecTypeDef::U128,
            ScSpecTypeDef::I128,
            ScSpecTypeDef::Symbol,
            ScSpecTypeDef::String,
            ScSpecTypeDef::Bytes,
        ];

        for element_type in element_types {
            let type_ = ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
                element_type: Box::new(element_type),
            }));

            // Test 0 count
            let val_zero = ramp_val(&env, &type_, 0).unwrap();
            let vec_zero: SVec<Val> = val_zero.into_val(&env);
            assert_eq!(vec_zero.len(), 0);

            // Test non-zero count
            let val = ramp_val(&env, &type_, 3).unwrap();
            let vec: SVec<Val> = val.into_val(&env);
            assert_eq!(vec.len(), 3);
        }
    }

    #[test]
    fn ramp_val_rejects_unsupported_scalar_and_composite_types() {
        let env = Env::default();

        let unsupported = [
            ScSpecTypeDef::Void,
            ScSpecTypeDef::Timepoint,
            ScSpecTypeDef::Duration,
        ];
        for ty in unsupported {
            let err = ramp_val(&env, &ty, 5).unwrap_err();
            assert!(err.0.contains("isn't supported yet"), "{}", err.0);
        }

        // Unsupported Vec element type
        let unsupported_vec = ScSpecTypeDef::Vec(Box::new(ScSpecTypeVec {
            element_type: Box::new(ScSpecTypeDef::Void),
        }));
        let err = ramp_val(&env, &unsupported_vec, 3).unwrap_err();
        assert!(
            err.0.contains("isn't supported as a ramp parameter yet"),
            "{}",
            err.0
        );
    }

    // ---- #497: measurement limitations in every output format ----

    #[test]
    fn export_results_json_includes_measurement_limitations_and_audit() {
        let dir = std::env::temp_dir().join(format!("stk-export-json-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");

        let audit = RefinementAudit {
            initial_low: 10,
            initial_high: 20,
            steps: vec![RefinementStep {
                step: 1,
                probe_value: 15,
                succeeded: true,
                failure_category: None,
                window_low: 15,
                window_high: 20,
            }],
            final_success: 15,
            final_failure: 16,
        };

        let summary = LimitsSummary {
            best: 15,
            instructions: 500_000,
            memory_bytes: 25_000,
            first_failed: Some(16),
            first_failure_category: Some(ProbeFailureCategory::ExitFailure),
            first_failed_probe_stderr: None,
            probe_failure_counts: BTreeMap::from([(ProbeFailureCategory::ExitFailure, 1)]),
            audit: Some(audit),
            ..sample_summary()
        };

        export_results(&path, &summary).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["contract"], "vault.wasm");
        assert_eq!(parsed["function"], "batch_payout");
        assert_eq!(parsed["ramp"], "recipients");
        assert_eq!(parsed["best"], 15);
        assert_eq!(parsed["instructions"], 500_000);
        assert_eq!(parsed["memory_bytes"], 25_000);
        assert_eq!(parsed["measurement_limitations"], MEASUREMENT_LIMITATIONS);
        assert_eq!(parsed["audit"]["steps"].as_array().unwrap().len(), 1);
        // The standardized failure categories appear in the JSON export.
        assert_eq!(parsed["first_failed"], 16);
        assert_eq!(parsed["first_failure_category"], "exit-failure");
        assert_eq!(parsed["probe_failure_counts"]["exit-failure"], 1);
    }

    #[test]
    fn render_text_reports_boundary_categories_and_unifies_with_json() {
        let summary = LimitsSummary {
            best: 15,
            instructions: 500_000,
            memory_bytes: 25_000,
            first_failed: Some(16),
            first_failure_category: Some(ProbeFailureCategory::TimedOut),
            first_failed_probe_stderr: None,
            probe_failure_counts: BTreeMap::from([
                (ProbeFailureCategory::ExitFailure, 2),
                (ProbeFailureCategory::TimedOut, 1),
            ]),
            audit: Some(RefinementAudit {
                initial_low: 10,
                initial_high: 20,
                steps: vec![RefinementStep {
                    step: 1,
                    probe_value: 15,
                    succeeded: false,
                    failure_category: Some(ProbeFailureCategory::TimedOut),
                    window_low: 10,
                    window_high: 15,
                }],
                final_success: 14,
                final_failure: 15,
            }),
            ..sample_summary()
        };

        let text = summary.render_text();
        assert!(text.contains("last successful recipients"), "{text}");
        assert!(
            text.contains("first failed recipients value: 16 [timed-out]"),
            "{text}"
        );
        assert!(
            text.contains("probe failures: exit-failure: 2, timed-out: 1"),
            "{text}"
        );
        assert!(
            text.contains("boundary refinement: probed 1 values"),
            "{text}"
        );
        assert!(text.contains("--probe-timeout (30s)"), "{text}");
        assert!(text.contains("reproduce locally:"), "{text}");

        // The same summary serializes the same boundary into JSON, so the
        // two formats cannot drift apart.
        let json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["best"], 15);
        assert_eq!(json["first_failure_category"], "timed-out");
        assert_eq!(json["probe_failure_counts"]["exit-failure"], 2);
        assert!(json["reproduction"]
            .as_str()
            .unwrap()
            .contains("batch_payout"));
    }

    #[test]
    fn build_reproduction_guidance_includes_args_and_clarity() {
        let guidance = build_reproduction_guidance(
            "vault.wasm",
            "batch_payout",
            "recipients",
            42,
            &["token=0x1234".to_string(), "mode=1".to_string()],
        );
        assert!(guidance.contains("invoke `batch_payout` with `recipients` = 42"));
        assert!(guidance.contains("--contract vault.wasm"));
        assert!(guidance.contains("--fn batch_payout"));
        assert!(guidance.contains("--ramp recipients"));
        assert!(guidance.contains("--arg token=0x1234 --arg mode=1"));
    }

    #[test]
    fn baseline_file_format_includes_measurement_limitations() {
        let dir = std::env::temp_dir().join(format!("stk-limitations-base-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.baseline");

        write_baseline(
            &path,
            Path::new("vault.wasm"),
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
        )
        .unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains(&format!(
            "measurement_limitations={MEASUREMENT_LIMITATIONS}"
        )));

        let parsed = Baseline::parse(&contents).unwrap();
        assert_eq!(
            parsed.measurement_limitations.as_deref(),
            Some(MEASUREMENT_LIMITATIONS)
        );
    }

    // ---- Bug: handle empty Bytes overrides without producing invalid XDR ----

    #[test]
    fn explicit_val_empty_bytes_override_produces_valid_empty_bytes() {
        let env = Env::default();
        // 0x with no hex digits â the regression input. Must produce the
        // same empty Bytes that `Bytes::new` does, not invalid XDR.
        let val = explicit_val(&env, &ScSpecTypeDef::Bytes, "0x").unwrap();
        let bytes: soroban_sdk::Bytes = val.into_val(&env);
        assert_eq!(bytes.len(), 0);
    }

    #[test]
    fn explicit_val_nonempty_bytes_override_still_works() {
        let env = Env::default();
        let val = explicit_val(&env, &ScSpecTypeDef::Bytes, "0xdeadbeef").unwrap();
        let bytes: soroban_sdk::Bytes = val.into_val(&env);
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn explicit_val_single_byte_bytes_override() {
        // Edge case: a one-byte (two hex digit) value.
        let env = Env::default();
        let val = explicit_val(&env, &ScSpecTypeDef::Bytes, "0x00").unwrap();
        let bytes: soroban_sdk::Bytes = val.into_val(&env);
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn explicit_val_bytes_override_rejects_missing_prefix() {
        let env = Env::default();
        let err = explicit_val(&env, &ScSpecTypeDef::Bytes, "deadbeef").unwrap_err();
        assert!(err.0.contains("must start with 0x"), "{}", err.0);
    }

    #[test]
    fn explicit_val_empty_bytes_matches_default_val() {
        // The empty override should produce the same empty Bytes that the
        // default path (which uses `Bytes::new`) produces.
        let env = Env::default();
        let override_val: soroban_sdk::Bytes = explicit_val(&env, &ScSpecTypeDef::Bytes, "0x")
            .unwrap()
            .into_val(&env);
        let default_val: soroban_sdk::Bytes = soroban_sdk::Bytes::new(&env).into_val(&env);
        assert_eq!(default_val.len(), 0);
        assert_eq!(override_val.len(), 0);
    }

    // ---- Bug: retain stderr context from a failed wasm probe ----

    #[test]
    fn limits_summary_renders_first_failed_probe_stderr() {
        let summary = LimitsSummary {
            first_failed: Some(10),
            first_failure_category: Some(ProbeFailureCategory::ExitFailure),
            first_failed_probe_stderr: Some("contract panicked: overflow".to_string()),
            ..sample_summary()
        };
        let text = summary.render_text();
        assert!(
            text.contains("first failing probe stderr: contract panicked: overflow"),
            "{text}"
        );
    }

    #[test]
    fn limits_summary_omits_stderr_line_when_empty() {
        let summary = LimitsSummary {
            first_failed_probe_stderr: Some("   \n  ".to_string()),
            ..sample_summary()
        };
        let text = summary.render_text();
        assert!(
            !text.contains("first failing probe stderr"),
            "whitespace-only stderr should not render: {text}"
        );
    }

    #[test]
    fn limits_summary_omits_stderr_line_when_none() {
        let summary = LimitsSummary {
            first_failed_probe_stderr: None,
            ..sample_summary()
        };
        let text = summary.render_text();
        assert!(!text.contains("first failing probe stderr"), "{text}");
    }

    #[test]
    fn limits_summary_includes_failed_probe_stderr_in_json() {
        let summary = LimitsSummary {
            first_failed_probe_stderr: Some("host abort".to_string()),
            ..sample_summary()
        };
        let json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["first_failed_probe_stderr"], "host abort");
    }

    #[test]
    fn limits_summary_omits_stderr_from_json_when_none() {
        let summary = LimitsSummary {
            first_failed_probe_stderr: None,
            ..sample_summary()
        };
        let json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        assert!(
            json.get("first_failed_probe_stderr").is_none()
                || json["first_failed_probe_stderr"].is_null(),
            "stderr should be skipped when None: {json}"
        );
    }

    // ---- Issue #233: paths with spaces ----

    #[test]
    fn sanitize_report_path_preserves_spaces_in_relative_paths() {
        let path = Path::new("path/to/my contract.wasm");
        let sanitized = sanitize_report_path(path);
        assert_eq!(sanitized, "path/to/my contract.wasm");
    }

    #[test]
    fn export_results_handles_path_with_spaces_json() {
        let dir = std::env::temp_dir().join(format!("stk-limits-spaces-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let export_path = dir.join("my results.json");

        let summary = sample_summary();
        let result = export_results(&export_path, &summary);
        assert!(
            result.is_ok(),
            "should handle spaces in export path: {:?}",
            result.err()
        );
        assert!(
            export_path.exists(),
            "export should create file at path with spaces"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn baseline_write_handles_path_with_spaces() {
        let dir = std::env::temp_dir().join(format!("stk-baseline-spaces-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let baseline_path = dir.join("my baseline file.baseline");

        let result = write_baseline(
            &baseline_path,
            Path::new("contract.wasm"),
            "recipients",
            "batch_payout",
            100,
            1_000_000,
            50_000,
        );
        assert!(
            result.is_ok(),
            "should handle spaces in baseline path: {:?}",
            result.err()
        );
        assert!(baseline_path.exists(), "baseline should be created");

        std::fs::remove_dir_all(&dir).ok();
    }
    // ---- #378: explain which resource boundary stopped the ramp ----

    #[test]
    fn resource_boundary_identifies_instruction_bound() {
        // 80M instructions out of 100M limit = 80%; 1MB out of 40MB = 2.5%
        let boundary = infer_resource_boundary(80_000_000, 1_000_000);
        assert_eq!(boundary, ResourceBoundary::Instructions);
        assert!(!boundary.explanation().is_empty());
    }

    #[test]
    fn resource_boundary_identifies_memory_bound() {
        // 1M instructions out of 100M = 1%; 35MB out of 40MB = 87.5%
        let boundary = infer_resource_boundary(1_000_000, 36_700_160);
        assert_eq!(boundary, ResourceBoundary::Memory);
    }

    #[test]
    fn resource_boundary_returns_other_when_both_low() {
        // Both below 25% threshold
        let boundary = infer_resource_boundary(1_000_000, 100_000);
        assert_eq!(boundary, ResourceBoundary::Other);
        assert!(boundary.explanation().contains("neither resource"));
    }

    #[test]
    fn resource_boundary_display_matches_serde() {
        assert_eq!(ResourceBoundary::Instructions.to_string(), "instructions");
        assert_eq!(ResourceBoundary::Memory.to_string(), "memory");
        assert_eq!(ResourceBoundary::Other.to_string(), "other");
        let json = serde_json::to_value(ResourceBoundary::Instructions).unwrap();
        assert_eq!(json, serde_json::json!("instructions"));
        let parsed: ResourceBoundary = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, ResourceBoundary::Instructions);
    }

    #[test]
    fn stopped_by_appears_in_render_text_when_set() {
        let summary = LimitsSummary {
            stopped_by: Some(ResourceBoundary::Instructions),
            ..sample_summary()
        };
        let text = summary.render_text();
        assert!(text.contains("resource boundary: instructions"), "{text}");
        assert!(text.contains("compute-bound"), "{text}");
    }

    #[test]
    fn stopped_by_appears_in_json_when_set() {
        let summary = LimitsSummary {
            stopped_by: Some(ResourceBoundary::Memory),
            ..sample_summary()
        };
        let json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["stopped_by"], "memory");
    }

    #[test]
    fn stopped_by_omitted_from_json_when_none() {
        let summary = LimitsSummary {
            stopped_by: None,
            ..sample_summary()
        };
        let json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        assert!(json.get("stopped_by").is_none() || json["stopped_by"].is_null());
    }

    // ---- #379: document how to interpret local instruction measurements ----

    #[test]
    fn instruction_interpretation_constant_is_non_empty_and_informative() {
        assert!(!INSTRUCTION_INTERPRETATION.is_empty());
        assert!(
            INSTRUCTION_INTERPRETATION.contains("100,000,000"),
            "{INSTRUCTION_INTERPRETATION}"
        );
        assert!(
            INSTRUCTION_INTERPRETATION.contains("mainnet"),
            "{INSTRUCTION_INTERPRETATION}"
        );
        assert!(
            INSTRUCTION_INTERPRETATION.contains("wall-clock"),
            "{INSTRUCTION_INTERPRETATION}"
        );
    }

    #[test]
    fn render_text_includes_instruction_interpretation() {
        let text = sample_summary().render_text();
        assert!(
            text.contains("instruction counts are Soroban host CPU instructions"),
            "render_text must include instruction interpretation: {text}"
        );
        assert!(
            text.contains("100,000,000"),
            "render_text must include mainnet limit: {text}"
        );
    }

    // ---- #377: named ramp presets ----

    #[test]
    fn ramp_preset_from_name_accepts_valid_presets() {
        assert_eq!(
            RampPreset::from_name("batch-payout").unwrap(),
            RampPreset::BatchPayout
        );
        assert_eq!(
            RampPreset::from_name("token-transfer").unwrap(),
            RampPreset::TokenTransfer
        );
        assert_eq!(
            RampPreset::from_name("multi-mint").unwrap(),
            RampPreset::MultiMint
        );
    }

    #[test]
    fn ramp_preset_from_name_rejects_unknown_preset() {
        let err = RampPreset::from_name("nonexistent-preset").unwrap_err();
        assert!(err.0.contains("unknown preset"), "{}", err.0);
        assert!(err.0.contains("batch-payout"), "{}", err.0);
        assert!(err.0.contains("token-transfer"), "{}", err.0);
        assert!(err.0.contains("multi-mint"), "{}", err.0);
    }

    #[test]
    fn ramp_preset_function_and_ramp_match_documented_values() {
        assert_eq!(RampPreset::BatchPayout.function(), "batch_payout");
        assert_eq!(RampPreset::BatchPayout.ramp(), "recipients");
        assert_eq!(RampPreset::TokenTransfer.function(), "transfer");
        assert_eq!(RampPreset::TokenTransfer.ramp(), "amount");
        assert_eq!(RampPreset::MultiMint.function(), "mint");
        assert_eq!(RampPreset::MultiMint.ramp(), "count");
    }

    fn make_preset_args(
        function: Option<&str>,
        ramp: Option<&str>,
        preset: Option<&str>,
    ) -> LimitsArgs {
        LimitsArgs {
            contract: std::path::PathBuf::from("x.wasm"),
            function: function.map(str::to_string),
            ramp: ramp.map(str::to_string),
            arg_values: vec![],
            probe_timeout: 30,
            timeout: None,
            progress: false,
            jobs: 1,
            baseline: None,
            baseline_tolerance_pct: 5.0,
            save_baseline: None,
            export: None,
            preset: preset.map(str::to_string),
            budget: None,
            co_ramp: Vec::new(),
            dry_run: false,
            export_samples: None,
            max_probes: 0,
            wasm_target_dir: None,
            profile: None,
            search_strategy: SearchStrategy::Doubling,
            step: 1,
            record_probe_duration: false,
        }
    }

    #[test]
    fn resolve_preset_defaults_uses_explicit_fn_over_preset() {
        let args = make_preset_args(Some("custom_fn"), None, Some("batch-payout"));
        let (function, ramp, _) = resolve_preset_defaults(&args).unwrap();
        assert_eq!(function, "custom_fn");
        assert_eq!(ramp, "recipients"); // falls back to preset
    }

    #[test]
    fn resolve_preset_defaults_uses_explicit_ramp_over_preset() {
        let args = make_preset_args(None, Some("custom_ramp"), Some("batch-payout"));
        let (function, ramp, _) = resolve_preset_defaults(&args).unwrap();
        assert_eq!(function, "batch_payout"); // falls back to preset
        assert_eq!(ramp, "custom_ramp");
    }

    #[test]
    fn resolve_preset_defaults_errors_without_preset_and_missing_fn() {
        let args = make_preset_args(None, Some("r"), None);
        let err = resolve_preset_defaults(&args).unwrap_err();
        assert!(err.0.contains("--fn is required"), "{}", err.0);
    }

    #[test]
    fn resolve_preset_defaults_errors_without_preset_and_missing_ramp() {
        let args = make_preset_args(Some("f"), None, None);
        let err = resolve_preset_defaults(&args).unwrap_err();
        assert!(err.0.contains("--ramp is required"), "{}", err.0);
    }

    // ---- #376: compare results against a checked-in resource budget ----

    #[test]
    fn resource_budget_parse_accepts_all_fields_with_comments() {
        let contents = "# budget for batch_payout
max_instructions=60000000
max_memory_bytes=20000000
min_best=50
";
        let budget = ResourceBudget::parse(contents).unwrap();
        assert_eq!(budget.max_instructions, Some(60_000_000));
        assert_eq!(budget.max_memory_bytes, Some(20_000_000));
        assert_eq!(budget.min_best, Some(50));
    }

    #[test]
    fn resource_budget_parse_accepts_partial_budget() {
        let budget = ResourceBudget::parse(
            "max_instructions=5000000
",
        )
        .unwrap();
        assert_eq!(budget.max_instructions, Some(5_000_000));
        assert!(budget.max_memory_bytes.is_none());
        assert!(budget.min_best.is_none());
    }

    #[test]
    fn resource_budget_parse_accepts_empty_file() {
        let budget = ResourceBudget::parse(
            "# only a comment

",
        )
        .unwrap();
        assert!(budget.max_instructions.is_none());
        assert!(budget.max_memory_bytes.is_none());
        assert!(budget.min_best.is_none());
    }

    #[test]
    fn resource_budget_parse_rejects_unknown_key() {
        let err = ResourceBudget::parse(
            "bogus_key=100
",
        )
        .unwrap_err();
        assert!(err.0.contains("unknown key"), "{}", err.0);
        assert!(err.0.contains("bogus_key"), "{}", err.0);
    }

    #[test]
    fn resource_budget_parse_rejects_malformed_line() {
        let err = ResourceBudget::parse(
            "not-key-value
",
        )
        .unwrap_err();
        assert!(err.0.contains("malformed budget file"), "{}", err.0);
    }

    #[test]
    fn resource_budget_parse_rejects_non_numeric_value() {
        let err = ResourceBudget::parse(
            "max_instructions=abc
",
        )
        .unwrap_err();
        assert!(err.0.contains("malformed budget"), "{}", err.0);
    }

    #[test]
    fn compare_to_budget_passes_when_within_all_limits() {
        let dir = std::env::temp_dir().join(format!("stk-budget-pass-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pass.budget");
        std::fs::write(
            &path,
            "max_instructions=2000000
max_memory_bytes=500000
min_best=50
",
        )
        .unwrap();
        let result = compare_to_budget(&path, "recipients", 100, 1_000_000, 200_000);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn compare_to_budget_fails_when_instructions_exceeded() {
        let dir = std::env::temp_dir().join(format!("stk-budget-insn-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("insn.budget");
        std::fs::write(
            &path,
            "max_instructions=500000
",
        )
        .unwrap();
        let err = compare_to_budget(&path, "recipients", 100, 1_000_000, 200_000).unwrap_err();
        assert!(err.0.contains("instructions 1000000 exceed"), "{}", err.0);
    }

    #[test]
    fn compare_to_budget_fails_when_memory_exceeded() {
        let dir = std::env::temp_dir().join(format!("stk-budget-mem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mem.budget");
        std::fs::write(
            &path,
            "max_memory_bytes=100000
",
        )
        .unwrap();
        let err = compare_to_budget(&path, "recipients", 100, 1_000_000, 200_000).unwrap_err();
        assert!(err.0.contains("memory bytes 200000 exceed"), "{}", err.0);
    }

    #[test]
    fn compare_to_budget_fails_when_ceiling_below_minimum() {
        let dir = std::env::temp_dir().join(format!("stk-budget-ceil-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ceil.budget");
        std::fs::write(
            &path,
            "min_best=200
",
        )
        .unwrap();
        let err = compare_to_budget(&path, "recipients", 100, 1_000_000, 200_000).unwrap_err();
        assert!(
            err.0
                .contains("recipients ceiling 100 is below the budget minimum of 200"),
            "{}",
            err.0
        );
    }

    #[test]
    fn compare_to_budget_reports_missing_file_actionably() {
        let missing = std::env::temp_dir().join("this-budget-does-not-exist.budget");
        let err = compare_to_budget(&missing, "recipients", 100, 1_000_000, 200_000).unwrap_err();
        assert!(
            err.0.contains("failed to read resource budget"),
            "{}",
            err.0
        );
    }

    #[test]
    fn compare_to_budget_reports_multiple_violations() {
        let dir = std::env::temp_dir().join(format!("stk-budget-multi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi.budget");
        std::fs::write(
            &path,
            "max_instructions=100
max_memory_bytes=100
min_best=200
",
        )
        .unwrap();
        let err = compare_to_budget(&path, "recipients", 50, 1_000_000, 200_000).unwrap_err();
        assert!(err.0.contains("resource budget exceeded"), "{}", err.0);
        // All three violations should be listed
        assert!(err.0.contains("instructions"), "{}", err.0);
        assert!(err.0.contains("memory"), "{}", err.0);
        assert!(err.0.contains("ceiling"), "{}", err.0);
    }

    // ── #372: dry-run mode ────────────────────────────────────────────────────

    #[test]
    fn dry_run_flag_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--dry-run",
        ])
        .expect("--dry-run should be accepted");
        assert!(args.dry_run);
    }

    #[test]
    fn dry_run_defaults_to_false() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
        ])
        .expect("parse should succeed");
        assert!(!args.dry_run);
    }

    // ── #373: export-samples ─────────────────────────────────────────────────

    #[test]
    fn export_samples_flag_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--export-samples",
            "/tmp/samples.json",
        ])
        .expect("--export-samples should be accepted");
        assert!(args.export_samples.is_some());
    }

    #[test]
    fn export_probe_samples_writes_json_array() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_probe_samples.json");
        let samples = vec![
            ProbeSample {
                value: 1,
                succeeded: true,
                failure_category: None,
                duration_ms: None,
            },
            ProbeSample {
                value: 2,
                succeeded: false,
                failure_category: Some(ProbeFailureCategory::ExitFailure),
                duration_ms: None,
            },
        ];
        export_probe_samples(&path, &samples).expect("export should succeed");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("\"value\": 1"),
            "missing value 1: {content}"
        );
        assert!(
            content.contains("\"succeeded\": true"),
            "missing succeeded true: {content}"
        );
        assert!(
            content.contains("\"failure_category\": \"exit-failure\""),
            "missing category: {content}"
        );
        let _ = std::fs::remove_file(&path);
    }

    // ── #368: co-ramp ────────────────────────────────────────────────────────

    #[test]
    fn co_ramp_flag_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--co-ramp",
            "m",
            "--co-ramp",
            "k",
        ])
        .expect("--co-ramp should be accepted");
        assert_eq!(args.co_ramp, vec!["m", "k"]);
    }

    #[test]
    fn build_args_with_co_ramp_index_applies_ramp_value() {
        let env = Env::new_with_config(soroban_sdk::testutils::EnvTestConfig {
            capture_snapshot_at_drop: false,
        });
        env.mock_all_auths_allowing_non_root_auth();
        let inputs = soroban_sdk::xdr::VecM::try_from(vec![
            soroban_sdk::xdr::ScSpecFunctionInputV0 {
                doc: soroban_sdk::xdr::StringM::default(),
                name: soroban_sdk::xdr::StringM::try_from(b"a".to_vec()).unwrap(),
                type_: soroban_sdk::xdr::ScSpecTypeDef::U32,
            },
            soroban_sdk::xdr::ScSpecFunctionInputV0 {
                doc: soroban_sdk::xdr::StringM::default(),
                name: soroban_sdk::xdr::StringM::try_from(b"b".to_vec()).unwrap(),
                type_: soroban_sdk::xdr::ScSpecTypeDef::U32,
            },
        ])
        .unwrap();
        let function_spec = soroban_sdk::xdr::ScSpecFunctionV0 {
            doc: soroban_sdk::xdr::StringM::default(),
            name: soroban_sdk::xdr::ScSymbol(
                soroban_sdk::xdr::StringM::try_from(b"f".to_vec()).unwrap(),
            ),
            inputs,
            outputs: soroban_sdk::xdr::VecM::default(),
        };
        // ramp index 0, co-ramp index 1, value 42
        let result = build_args(&env, &function_spec, 0, &[1], 42, &[]).unwrap();
        assert_eq!(result.len(), 2);
    }

    // ── #369: max-probes ─────────────────────────────────────────────────────

    #[test]
    fn max_probes_flag_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--max-probes",
            "10",
        ])
        .expect("--max-probes should be accepted");
        assert_eq!(args.max_probes, 10);
    }

    #[test]
    fn max_probes_defaults_to_zero() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
        ])
        .expect("parse should succeed");
        assert_eq!(args.max_probes, 0);
    }

    // ── #374: compiler and target metadata ───────────────────────────────────

    #[test]
    fn collect_compiler_meta_returns_non_empty_rustc_version() {
        let meta = collect_compiler_meta();
        assert!(
            !meta.rustc_version.is_empty(),
            "rustc_version should not be empty"
        );
        assert_eq!(meta.wasm_target, "wasm32v1-none");
    }

    #[test]
    fn compiler_meta_serializes_with_kebab_case_keys() {
        let meta = CompilerMeta {
            rustc_version: "rustc 1.80.0 (051478957 2024-07-21)".to_string(),
            wasm_target: "wasm32v1-none".to_string(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(
            json.contains("\"rustc-version\""),
            "expected kebab-case key: {json}"
        );
        assert!(
            json.contains("\"wasm-target\""),
            "expected kebab-case key: {json}"
        );
    }

    #[test]
    fn limits_summary_compiler_meta_field_round_trips() {
        let meta = CompilerMeta {
            rustc_version: "rustc 1.80.0".to_string(),
            wasm_target: "wasm32v1-none".to_string(),
        };
        let summary = LimitsSummary {
            contract: "test.wasm".to_string(),
            function: Some("f".to_string()),
            ramp: Some("n".to_string()),
            best: 10,
            instructions: 1000,
            memory_bytes: 2000,
            first_failed: None,
            first_failure_category: None,
            first_failed_probe_stderr: None,
            last_failed_probe: None,
            probe_failure_counts: BTreeMap::new(),
            hit_upper_bound: false,
            probe_timeout_secs: 30,
            audit: None,
            measurement_limitations: MEASUREMENT_LIMITATIONS.to_string(),
            reproduction: None,
            stopped_by: None,
            probe_samples: vec![],
            co_ramp: vec![],
            compiler_meta: Some(meta.clone()),
            wasm_target_dir: None,
            profile: None,
        };
        let json = serde_json::to_string_pretty(&summary).unwrap();
        assert!(
            json.contains("rustc-version"),
            "compiler-meta missing from JSON: {json}"
        );
        let decoded: LimitsSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.compiler_meta, Some(meta));
    }

    #[test]
    fn render_text_includes_compiler_meta() {
        let summary = LimitsSummary {
            contract: "test.wasm".to_string(),
            function: Some("f".to_string()),
            ramp: Some("n".to_string()),
            best: 10,
            instructions: 1000,
            memory_bytes: 2000,
            first_failed: None,
            first_failure_category: None,
            first_failed_probe_stderr: None,
            last_failed_probe: None,
            probe_failure_counts: BTreeMap::new(),
            hit_upper_bound: false,
            probe_timeout_secs: 30,
            audit: None,
            measurement_limitations: MEASUREMENT_LIMITATIONS.to_string(),
            reproduction: None,
            stopped_by: None,
            probe_samples: vec![],
            co_ramp: vec![],
            compiler_meta: Some(CompilerMeta {
                rustc_version: "rustc 1.80.0".to_string(),
                wasm_target: "wasm32v1-none".to_string(),
            }),
            wasm_target_dir: None,
            profile: None,
        };
        let text = summary.render_text();
        assert!(
            text.contains("rustc 1.80.0"),
            "compiler version missing from text: {text}"
        );
        assert!(
            text.contains("wasm32v1-none"),
            "wasm target missing from text: {text}"
        );
    }

    // ── #370: custom wasm target directory ───────────────────────────────────

    #[test]
    fn wasm_target_dir_flag_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--wasm-target-dir",
            "/custom/target",
        ])
        .expect("--wasm-target-dir should be accepted");
        assert!(args.wasm_target_dir.is_some());
        assert_eq!(
            args.wasm_target_dir.unwrap().to_str().unwrap(),
            "/custom/target"
        );
    }

    #[test]
    fn wasm_target_dir_defaults_to_none() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
        ])
        .expect("parse should succeed");
        assert!(args.wasm_target_dir.is_none());
    }

    #[test]
    fn render_text_includes_wasm_target_dir_when_set() {
        let summary = LimitsSummary {
            contract: "test.wasm".to_string(),
            function: Some("f".to_string()),
            ramp: Some("n".to_string()),
            best: 10,
            instructions: 1000,
            memory_bytes: 2000,
            first_failed: None,
            first_failure_category: None,
            first_failed_probe_stderr: None,
            last_failed_probe: None,
            probe_failure_counts: BTreeMap::new(),
            hit_upper_bound: false,
            probe_timeout_secs: 30,
            audit: None,
            measurement_limitations: MEASUREMENT_LIMITATIONS.to_string(),
            reproduction: None,
            stopped_by: None,
            probe_samples: vec![],
            co_ramp: vec![],
            compiler_meta: None,
            wasm_target_dir: Some("/custom/target".to_string()),
            profile: None,
        };
        let text = summary.render_text();
        assert!(
            text.contains("/custom/target"),
            "wasm_target_dir missing from text: {text}"
        );
    }

    // ── #371: cargo profile selection ────────────────────────────────────────

    #[test]
    fn profile_flag_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--profile",
            "dev",
        ])
        .expect("--profile should be accepted");
        assert_eq!(args.profile.as_deref(), Some("dev"));
    }

    #[test]
    fn profile_defaults_to_none() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
        ])
        .expect("parse should succeed");
        assert!(args.profile.is_none());
    }

    #[test]
    fn render_text_includes_profile_when_set() {
        let summary = LimitsSummary {
            contract: "test.wasm".to_string(),
            function: Some("f".to_string()),
            ramp: Some("n".to_string()),
            best: 10,
            instructions: 1000,
            memory_bytes: 2000,
            first_failed: None,
            first_failure_category: None,
            first_failed_probe_stderr: None,
            last_failed_probe: None,
            probe_failure_counts: BTreeMap::new(),
            hit_upper_bound: false,
            probe_timeout_secs: 30,
            audit: None,
            measurement_limitations: MEASUREMENT_LIMITATIONS.to_string(),
            reproduction: None,
            stopped_by: None,
            probe_samples: vec![],
            co_ramp: vec![],
            compiler_meta: None,
            wasm_target_dir: None,
            profile: Some("dev".to_string()),
        };
        let text = summary.render_text();
        assert!(text.contains("dev"), "profile missing from text: {text}");
    }

    // ── #375: reusable baseline comparison ───────────────────────────────────

    #[test]
    fn baseline_comparison_new_detects_no_regression() {
        let baseline = Baseline {
            contract: None,
            function: "f".to_string(),
            ramp: "n".to_string(),
            best: 100,
            instructions: 50_000,
            memory_bytes: 10_000,
            measurement_limitations: None,
        };
        let cmp = BaselineComparison::new(&baseline, "n", "f", 100, 50_000, 10_000, 5.0).unwrap();
        assert!(!cmp.has_regression());
        assert!(cmp.regressions().is_empty());
    }

    #[test]
    fn baseline_comparison_detects_ceiling_regression() {
        let baseline = Baseline {
            contract: None,
            function: "f".to_string(),
            ramp: "n".to_string(),
            best: 100,
            instructions: 50_000,
            memory_bytes: 10_000,
            measurement_limitations: None,
        };
        let cmp = BaselineComparison::new(&baseline, "n", "f", 90, 50_000, 10_000, 5.0).unwrap();
        assert!(cmp.has_regression());
        assert_eq!(cmp.regressions().len(), 1);
        assert!(
            cmp.regressions()[0].contains("ceiling dropped"),
            "{}",
            cmp.regressions()[0]
        );
    }

    #[test]
    fn baseline_comparison_detects_instruction_regression_beyond_tolerance() {
        let baseline = Baseline {
            contract: None,
            function: "f".to_string(),
            ramp: "n".to_string(),
            best: 100,
            instructions: 50_000,
            memory_bytes: 10_000,
            measurement_limitations: None,
        };
        let cmp = BaselineComparison::new(&baseline, "n", "f", 100, 60_000, 10_000, 5.0).unwrap();
        assert!(cmp.has_regression());
        assert!(
            cmp.regressions()[0].contains("instructions"),
            "{}",
            cmp.regressions()[0]
        );
    }

    #[test]
    fn baseline_comparison_allows_instruction_growth_within_tolerance() {
        let baseline = Baseline {
            contract: None,
            function: "f".to_string(),
            ramp: "n".to_string(),
            best: 100,
            instructions: 50_000,
            memory_bytes: 10_000,
            measurement_limitations: None,
        };
        let cmp = BaselineComparison::new(&baseline, "n", "f", 100, 52_000, 10_000, 5.0).unwrap();
        assert!(!cmp.has_regression());
    }

    #[test]
    fn baseline_comparison_errors_on_function_mismatch() {
        let baseline = Baseline {
            contract: None,
            function: "other_fn".to_string(),
            ramp: "n".to_string(),
            best: 100,
            instructions: 50_000,
            memory_bytes: 10_000,
            measurement_limitations: None,
        };
        let err =
            BaselineComparison::new(&baseline, "n", "f", 100, 50_000, 10_000, 5.0).unwrap_err();
        assert!(err.0.contains("other_fn"), "{}", err.0);
    }

    #[test]
    fn baseline_comparison_carries_all_metrics() {
        let baseline = Baseline {
            contract: None,
            function: "f".to_string(),
            ramp: "n".to_string(),
            best: 100,
            instructions: 50_000,
            memory_bytes: 10_000,
            measurement_limitations: None,
        };
        let cmp = BaselineComparison::new(&baseline, "n", "f", 110, 55_000, 11_000, 10.0).unwrap();
        assert_eq!(cmp.baseline_best, 100);
        assert_eq!(cmp.current_best, 110);
        assert_eq!(cmp.baseline_instructions, 50_000);
        assert_eq!(cmp.current_instructions, 55_000);
        assert_eq!(cmp.baseline_memory_bytes, 10_000);
        assert_eq!(cmp.current_memory_bytes, 11_000);
    }

    // ---- #364: probe duration recording ----

    #[test]
    fn probe_sample_duration_ms_is_absent_by_default() {
        let sample = ProbeSample {
            value: 10,
            succeeded: true,
            failure_category: None,
            duration_ms: None,
        };
        let json = serde_json::to_string(&sample).unwrap();
        assert!(
            !json.contains("duration_ms"),
            "duration_ms should be absent when None: {json}"
        );
    }

    #[test]
    fn probe_sample_duration_ms_is_present_when_set() {
        let sample = ProbeSample {
            value: 10,
            succeeded: true,
            failure_category: None,
            duration_ms: Some(42),
        };
        let json = serde_json::to_string(&sample).unwrap();
        assert!(
            json.contains("\"duration_ms\":42"),
            "duration_ms missing: {json}"
        );
    }

    #[test]
    fn probe_sample_duration_ms_round_trips_through_json() {
        let sample = ProbeSample {
            value: 7,
            succeeded: false,
            failure_category: Some(ProbeFailureCategory::TimedOut),
            duration_ms: Some(1234),
        };
        let json = serde_json::to_string(&sample).unwrap();
        let back: ProbeSample = serde_json::from_str(&json).unwrap();
        assert_eq!(back.duration_ms, Some(1234));
        assert_eq!(back.failure_category, Some(ProbeFailureCategory::TimedOut));
    }

    // ---- #365: search strategy selection ----

    #[test]
    fn search_strategy_flag_accepts_doubling() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--search-strategy",
            "doubling",
        ])
        .expect("--search-strategy doubling should be accepted");
        assert_eq!(args.search_strategy, SearchStrategy::Doubling);
    }

    #[test]
    fn search_strategy_flag_accepts_fixed_step() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--search-strategy",
            "fixed-step",
        ])
        .expect("--search-strategy fixed-step should be accepted");
        assert_eq!(args.search_strategy, SearchStrategy::FixedStep);
    }

    #[test]
    fn search_strategy_defaults_to_doubling() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
        ])
        .expect("parse should succeed");
        assert_eq!(args.search_strategy, SearchStrategy::Doubling);
    }

    #[test]
    fn search_strategy_rejects_unknown_value() {
        let err = SearchStrategy::from_str("exponential").unwrap_err();
        assert!(err.0.contains("unknown search strategy"), "{}", err.0);
        assert!(err.0.contains("doubling"), "{}", err.0);
        assert!(err.0.contains("fixed-step"), "{}", err.0);
    }

    // ---- #366: fixed-step ramp mode ----

    #[test]
    fn fixed_step_batch_produces_multiples_of_step() {
        let batch = fixed_step_batch(10, 100, 10, 5);
        assert_eq!(batch, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn fixed_step_batch_clamps_at_upper_bound() {
        let batch = fixed_step_batch(90, 100, 10, 5);
        assert_eq!(batch, vec![90, 100]);
    }

    #[test]
    fn fixed_step_batch_stops_when_at_upper_bound() {
        let batch = fixed_step_batch(100, 100, 10, 5);
        assert_eq!(batch, vec![100]);
    }

    #[test]
    fn next_fixed_step_value_advances_by_step() {
        assert_eq!(next_fixed_step_value(10, 100, 10), Some(20));
        assert_eq!(next_fixed_step_value(90, 100, 10), Some(100));
        assert_eq!(next_fixed_step_value(100, 100, 10), None);
    }

    #[test]
    fn step_flag_is_accepted_by_limits_args() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--search-strategy",
            "fixed-step",
            "--step",
            "50",
        ])
        .expect("--step should be accepted");
        assert_eq!(args.step, 50);
        assert_eq!(args.search_strategy, SearchStrategy::FixedStep);
    }

    // ---- #367: signed integer ramp parameter support ----

    #[test]
    fn ramp_val_i32_produces_expected_positive_sequence() {
        let env = Env::default();
        for v in [0u32, 1, 100, 1000, i32::MAX as u32] {
            let val = ramp_val(&env, &ScSpecTypeDef::I32, v).unwrap();
            let got: i32 = val.into_val(&env);
            assert_eq!(got, v as i32, "i32 cast failed for {v}");
        }
    }

    #[test]
    fn ramp_val_i64_produces_expected_positive_sequence() {
        let env = Env::default();
        for v in [0u32, 1, 1_000_000, u32::MAX] {
            let val = ramp_val(&env, &ScSpecTypeDef::I64, v).unwrap();
            let got: i64 = val.into_val(&env);
            assert_eq!(got, v as i64, "i64 cast failed for {v}");
        }
    }

    #[test]
    fn ramp_val_i128_produces_expected_positive_sequence() {
        let env = Env::default();
        for v in [0u32, 1, 1_000_000, u32::MAX] {
            let val = ramp_val(&env, &ScSpecTypeDef::I128, v).unwrap();
            let got: i128 = val.into_val(&env);
            assert_eq!(got, v as i128, "i128 cast failed for {v}");
        }
    }

    #[test]
    fn record_probe_duration_flag_defaults_to_false() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
        ])
        .expect("parse should succeed");
        assert!(!args.record_probe_duration);
    }

    #[test]
    fn record_probe_duration_flag_is_accepted() {
        let args = LimitsArgs::try_parse_from([
            "limits",
            "--contract",
            "contract.wasm",
            "--fn",
            "test",
            "--ramp",
            "n",
            "--record-probe-duration",
        ])
        .expect("--record-probe-duration should be accepted");
        assert!(args.record_probe_duration);
    }

    // ---- #360: per-probe timeout ----

    #[test]
    fn timed_out_note_names_the_ramp_value_and_the_timeout() {
        let note = timed_out_note("recipients", 64, Duration::from_secs(30));
        assert!(note.contains("recipients=64"), "{note}");
        assert!(note.contains("30s timeout"), "{note}");
        assert!(note.contains("raise --probe-timeout"), "{note}");
    }

    #[test]
    fn format_duration_uses_the_shortest_readable_unit() {
        assert_eq!(format_duration(Duration::ZERO), "0ms");
        assert_eq!(format_duration(Duration::from_millis(250)), "250ms");
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_millis(1_500)), "1.5s");
    }

    #[test]
    fn probe_timeout_that_overflows_the_clock_is_rejected_before_probing() {
        let err = validate_timeouts(u64::MAX, None).unwrap_err();
        assert!(err.0.contains("--probe-timeout"), "{}", err.0);
        assert!(err.0.contains("too large for the host clock"), "{}", err.0);
    }

    #[test]
    fn realistic_probe_timeouts_are_accepted() {
        // 0 is kept: it gives an immediate deadline, which tests rely on.
        for secs in [0, 1, 30, 86_400] {
            validate_timeouts(secs, None).unwrap();
        }
    }

    #[test]
    fn probe_timeout_flag_parses_and_defaults_to_thirty_seconds() {
        let base = ["limits", "--contract", "c.wasm", "--fn", "f", "--ramp", "n"];
        assert_eq!(LimitsArgs::try_parse_from(base).unwrap().probe_timeout, 30);
        let args = LimitsArgs::try_parse_from(base.iter().copied().chain(["--probe-timeout", "5"]))
            .unwrap();
        assert_eq!(args.probe_timeout, 5);
        assert!(
            LimitsArgs::try_parse_from(base.iter().copied().chain(["--probe-timeout", "-1"]))
                .is_err()
        );
    }

    // ---- #361: overall command timeout ----

    #[test]
    fn overall_timeout_must_be_positive_and_fit_the_clock() {
        let zero = validate_timeouts(30, Some(0)).unwrap_err();
        assert!(
            zero.0.contains("--timeout must be at least 1 second"),
            "{}",
            zero.0
        );
        let huge = validate_timeouts(30, Some(u64::MAX)).unwrap_err();
        assert!(huge.0.contains("--timeout"), "{}", huge.0);
        assert!(
            huge.0.contains("too large for the host clock"),
            "{}",
            huge.0
        );
        validate_timeouts(30, Some(1)).unwrap();
        validate_timeouts(30, Some(600)).unwrap();
    }

    #[test]
    fn timeout_flag_is_optional() {
        let base = ["limits", "--contract", "c.wasm", "--fn", "f", "--ramp", "n"];
        assert_eq!(LimitsArgs::try_parse_from(base).unwrap().timeout, None);
        let args =
            LimitsArgs::try_parse_from(base.iter().copied().chain(["--timeout", "120"])).unwrap();
        assert_eq!(args.timeout, Some(120));
    }

    #[test]
    fn without_an_overall_timeout_probes_get_the_full_per_probe_timeout() {
        let clock = RunClock::start(None);
        assert_eq!(
            clock.probe_wait(Duration::from_secs(30)),
            Duration::from_secs(30)
        );
        clock.check(5, Some(3), "f", "n").unwrap();
    }

    #[test]
    fn probe_wait_is_clamped_to_the_time_left() {
        let clock = RunClock::start(Some(Duration::from_secs(2)));
        let wait = clock.probe_wait(Duration::from_secs(30));
        assert!(wait <= Duration::from_secs(2), "{wait:?}");
        // A per-probe timeout shorter than the time left is used as-is.
        assert_eq!(
            clock.probe_wait(Duration::from_millis(10)),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn an_expired_deadline_reports_progress_and_how_to_recover() {
        let clock = RunClock::start(Some(Duration::ZERO));
        let err = clock
            .check(7, Some(16), "batch_payout", "recipients")
            .unwrap_err();
        assert!(
            err.0.contains("\"batch_payout\" exceeded --timeout (0ms)"),
            "{}",
            err.0
        );
        assert!(err.0.contains("after 7 probe(s)"), "{}", err.0);
        assert!(
            err.0.contains("highest recipients verified so far is 16"),
            "{}",
            err.0
        );
        assert!(err.0.contains("boundary was not confirmed"), "{}", err.0);
        assert!(err.0.contains("Raise --timeout"), "{}", err.0);

        let none = clock.check(1, None, "f", "n").unwrap_err();
        assert!(
            none.0.contains("no n value had succeeded yet"),
            "{}",
            none.0
        );
    }

    #[test]
    fn a_probe_cut_short_by_the_deadline_ends_the_run_instead_of_counting_as_a_ceiling() {
        let clock = RunClock::start(None);
        let timed_out = ProbeOutcome::Failed(ProbeFailureCategory::TimedOut);
        let err = clock
            .check_cut_short(
                timed_out,
                Duration::from_secs(1),
                Duration::from_secs(30),
                9,
                4,
                Some(8),
                "f",
                "n",
            )
            .unwrap_err();
        assert!(err.0.contains("while probing n=9"), "{}", err.0);
        assert!(
            err.0.contains("highest n verified so far is 8"),
            "{}",
            err.0
        );
    }

    #[test]
    fn a_probe_that_used_its_full_timeout_is_a_genuine_timeout() {
        let clock = RunClock::start(None);
        let full = Duration::from_secs(30);
        let timed_out = ProbeOutcome::Failed(ProbeFailureCategory::TimedOut);
        clock
            .check_cut_short(timed_out, full, full, 9, 4, Some(8), "f", "n")
            .unwrap();
        // Other outcomes are never treated as the deadline, even when clamped.
        for outcome in [
            ProbeOutcome::Succeeded,
            ProbeOutcome::Failed(ProbeFailureCategory::ExitFailure),
        ] {
            clock
                .check_cut_short(
                    outcome,
                    Duration::from_secs(1),
                    full,
                    9,
                    4,
                    Some(8),
                    "f",
                    "n",
                )
                .unwrap();
        }
    }

    // ---- #362: progress updates ----

    #[test]
    fn progress_line_for_a_successful_scan_probe() {
        let line = format_progress_line(
            3,
            ProbePhase::Scan,
            "recipients",
            4,
            ProbeOutcome::Succeeded,
            Some(4),
            Duration::from_millis(1_500),
        );
        assert_eq!(
            line,
            "limits: probe 3 (scan) recipients=4 ok; best so far: 4; elapsed 1.5s"
        );
    }

    #[test]
    fn progress_line_for_a_failed_refinement_probe() {
        let line = format_progress_line(
            9,
            ProbePhase::Refine,
            "recipients",
            48,
            ProbeOutcome::Failed(ProbeFailureCategory::TimedOut),
            Some(32),
            Duration::from_secs(12),
        );
        assert_eq!(
            line,
            "limits: probe 9 (refine) recipients=48 failed [timed-out]; best so far: 32; elapsed 12s"
        );
    }

    #[test]
    fn progress_line_before_any_success() {
        let line = format_progress_line(
            1,
            ProbePhase::Scan,
            "n",
            1,
            ProbeOutcome::Failed(ProbeFailureCategory::ExitFailure),
            None,
            Duration::ZERO,
        );
        assert!(line.contains("failed [exit-failure]"), "{line}");
        assert!(line.contains("best so far: none yet"), "{line}");
    }

    #[test]
    fn progress_flag_defaults_to_off() {
        let base = ["limits", "--contract", "c.wasm", "--fn", "f", "--ramp", "n"];
        assert!(!LimitsArgs::try_parse_from(base).unwrap().progress);
        assert!(
            LimitsArgs::try_parse_from(base.iter().copied().chain(["--progress"]))
                .unwrap()
                .progress
        );
    }

    // ---- #363: last failing probe ----

    fn summary_with_last_failure(stderr: Option<&str>) -> LimitsSummary {
        LimitsSummary {
            last_failed_probe: Some(FailedProbe {
                value: 101,
                category: ProbeFailureCategory::ExitFailure,
                stderr: stderr.map(str::to_string),
            }),
            ..sample_summary()
        }
    }

    #[test]
    fn text_summary_reports_the_last_failing_probe_next_to_the_maximum_success() {
        let text = summary_with_last_failure(Some("HostError: budget exceeded")).render_text();
        let lines: Vec<&str> = text.lines().collect();
        let best = lines
            .iter()
            .position(|l| l.starts_with("last successful"))
            .unwrap();
        let last = lines
            .iter()
            .position(|l| l.contains("last failing probe:"))
            .expect("last failing probe line missing");
        assert!(last > best && last <= best + 4, "{text}");
        assert_eq!(
            lines[last],
            "  last failing probe: recipients=101 [exit-failure]: HostError: budget exceeded"
        );
    }

    #[test]
    fn last_failing_probe_line_keeps_only_the_first_stderr_line() {
        let text =
            summary_with_last_failure(Some("\n  panicked at host.rs\nstack backtrace:\n  0: ..."))
                .render_text();
        assert!(
            text.contains("recipients=101 [exit-failure]: panicked at host.rs\n"),
            "{text}"
        );
        assert!(!text.contains("stack backtrace"), "{text}");
    }

    #[test]
    fn last_failing_probe_without_stderr_still_reports_value_and_category() {
        for stderr in [None, Some("   ")] {
            let text = summary_with_last_failure(stderr).render_text();
            assert!(
                text.contains("  last failing probe: recipients=101 [exit-failure]\n"),
                "{text}"
            );
        }
    }

    #[test]
    fn no_last_failing_probe_line_when_nothing_failed() {
        let text = sample_summary().render_text();
        assert!(!text.contains("last failing probe"), "{text}");
    }

    #[test]
    fn last_failing_probe_round_trips_through_json_and_old_reports_still_parse() {
        let summary = summary_with_last_failure(Some("budget exceeded"));
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["last_failed_probe"]["value"], 101);
        assert_eq!(json["last_failed_probe"]["category"], "exit-failure");
        assert_eq!(json["last_failed_probe"]["stderr"], "budget exceeded");
        let back: LimitsSummary = serde_json::from_value(json).unwrap();
        assert_eq!(back, summary);

        // Reports written before this field existed omit it entirely.
        let mut old = serde_json::to_value(sample_summary()).unwrap();
        old.as_object_mut().unwrap().remove("last_failed_probe");
        let parsed: LimitsSummary = serde_json::from_value(old).unwrap();
        assert_eq!(parsed.last_failed_probe, None);
    }
}
