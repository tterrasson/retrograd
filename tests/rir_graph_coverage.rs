//! RIR coverage on a **real** training graph (phase D).
//!
//! `tests/rir_probe.rs` proves the variant runs for one op at a time, on shapes
//! the probe itself builds. That is not the same claim as "the graph a model
//! actually trains does not fall back": the shapes a probe can express are
//! contiguous by construction, while the graph sends views inside packed
//! tensors. Phase D's measurement - 720 nodes, 0 eligible - was invisible to
//! the probe and only appeared here.
//!
//! So this test loads a model, trains a step, and reads the counters. A single
//! native fallback for a reason the registry does **not** declare is a failure:
//! the coverage number published in the plan must be a test, not a one-off
//! measurement someone re-ran by hand.
//!
//! It does not demand `native == 0`, which only holds while every integrated op
//! covers its ggml op entirely: `OUT_PROD` and `MUL` both leave legitimate ggml
//! the kernel never claimed. The criterion is the *declared* domain
//! (`rir_op_policy.assumed_domain`): a fallback inside
//! it is the published restriction working, a fallback outside it is a node the
//! kernel claimed and did not serve. That declaration is per *category*, so a
//! narrowing inside one it already names is caught by the claimed-node baseline
//! in `scripts/rir-domain-baseline.tsv` instead, not here.
//!
//! It needs three things and skips (loudly) without any of them: a GPU backend,
//! `RETRO_RIR_MODE=prefer` in the environment before the process starts (the
//! policy is fixed before the backend context exists), and a model whose
//! backward graph contains an op with a RIR variant. `RETRO_RIR_TEST_MODEL`
//! overrides the default path.

mod common;

use retrograd::{
    Device, LoraConfig, TargetSet, TrainConfig, Trainer, rir_census_report, rir_counters,
};

/// No default path: `RETRO_RIR_TEST_MODEL` must name a model whose backward
/// graph carries an op with a RIR variant. Any of the six integrated ops will
/// do - the assertions read whatever sites the graph produced instead of naming
/// one, so a model without QK-norm measures its own ops rather than passing
/// vacuously on an op it never emits.
const MODEL_ENV: &str = "RETRO_RIR_TEST_MODEL";

const TRAIN_TEXT: &str = concat!(
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump! ",
    "Sphinx of black quartz, judge my vow. ",
);

fn model_path() -> Option<std::path::PathBuf> {
    std::env::var(MODEL_ENV).ok().map(Into::into)
}

/// Whether this run is `scripts/test-rir-graph.sh` rather than someone's
/// terminal. A lane that skips is a lane that lies: every reason this test has
/// to step aside is a missing precondition the lane *provides*, so under
/// `RETRO_REQUIRE_RIR_GRAPH=1` each of them becomes a failure. The same shape
/// as `RETRO_REQUIRE_CPU_FIXTURE`.
fn required() -> bool {
    std::env::var("RETRO_REQUIRE_RIR_GRAPH").as_deref() == Ok("1")
}

/// Skip, unless the lane asked for the measurement - then fail with the same
/// sentence, so what a reader has to fix is identical in both modes.
macro_rules! skip_or_fail {
    ($($arg:tt)*) => {{
        if required() {
            panic!($($arg)*);
        }
        eprintln!("skipped: {}", format!($($arg)*));
        return;
    }};
}

/// Whether RIR may encode in this process. See [`common::rir_encoding_enabled`]
/// for why an unset variable reads as `prefer` rather than `off`.
fn rir_enabled() -> bool {
    common::rir_encoding_enabled()
}

#[test]
fn the_real_graph_never_falls_back_silently_under_prefer() {
    if !rir_enabled() {
        skip_or_fail!("RETRO_RIR_MODE lowers RIR below prefer; graph coverage measures nothing");
    }
    if !common::gpu_device_present() {
        skip_or_fail!("no GPU device registered at runtime");
    }
    let Some(model) = model_path() else {
        skip_or_fail!("set {MODEL_ENV} to a model whose graph exercises a RIR op");
    };
    assert!(
        model.exists(),
        "{MODEL_ENV} points at {} which does not exist",
        model.display()
    );

    let config = TrainConfig {
        n_ctx: 128,
        n_batch: 64,
        n_ubatch: 16,
        epochs: 1,
        learning_rate: 1.0e-4,
        device: Device::Gpu,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(&model, config).expect("load gpu trainer");
    let mut lora = LoraConfig::qv(1, 2.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(vec!["blk.*.attn_q.weight".to_string()]);
    trainer.create_lora(&lora).expect("create lora");

    let before = rir_counters().expect("rir_counters");
    let tokens = trainer
        .tokenize_text(&TRAIN_TEXT.repeat(12))
        .expect("tokenize");
    let metrics = trainer.train_tokens(&tokens).expect("train one epoch");
    assert!(metrics.train_loss.is_finite(), "train_loss must be finite");
    let after = rir_counters().expect("rir_counters");

    let seen = after.ops_seen - before.ops_seen;
    let dispatched = after.rir_dispatched - before.rir_dispatched;
    let native = after.native_dispatched - before.native_dispatched;
    eprintln!("rir graph coverage: seen={seen} rir={dispatched} native={native}");

    if seen == 0 {
        skip_or_fail!(
            "{} emits no op with a RIR variant - the measurement needs \
             a model whose backward graph does",
            model.display()
        );
    }
    // The coverage, as the trainer reports it. A number
    // that only exists in a test is a number nobody looks at; this is the one an
    // operator reads, so it has to carry the same rows and it has to be live
    // rather than served from the report cache.
    let report = trainer.capability_report().expect("capability_report");
    assert!(
        report.contains("  rir:\n"),
        "capability_report must carry the RIR section:\n{report}"
    );
    assert!(
        report.contains("      rir_dispatched: 0\n") == (dispatched == 0),
        "the report's trainer-scoped counters disagree with the FFI counters:\n{report}"
    );

    let sites = Site::parse(&report);
    assert!(
        !sites.is_empty(),
        "capability_report must break the counters down by op and backend:\n{report}"
    );

    // What replaced `native == 0`.
    //
    // That assertion was right only while every integrated op covered its ggml
    // op entirely. It cannot survive `OUT_PROD`, whose src0 is a quantized
    // frozen weight on three quarters of a LoRA backward's nodes, nor `MUL`,
    // whose src1 is broadcast on a third of them - both legitimate ggml, both
    // outside what the kernel claims, and both answered by the native kernel
    // exactly as intended. Demanding zero there would forbid integrating any
    // partial-domain op, which from here on is most of them.
    //
    // So the criterion moved from *how much* fell back to *whether the registry
    // said it would*. `assumed_domain` publishes the parts of the op a kernel
    // does not claim; a fallback inside it is the declared restriction working,
    // and a fallback outside it is a node the kernel claimed and did not serve.
    // The second is a regression, and it is now the only thing that fails here.
    for site in &sites {
        eprintln!(
            "rir graph coverage: {:<24} {:<7} {}/{} nodes ({} %) rejects=[{}] domain=[{}]",
            site.op,
            site.backend,
            site.rir,
            site.seen,
            // A site with no node seen reports 0 %, not a division by zero.
            (100 * site.rir).checked_div(site.seen).unwrap_or(0),
            site.reject_summary(),
            if site.domain.is_empty() {
                "none".to_string()
            } else {
                site.domain.join("|")
            },
        );
        if site.retired {
            eprintln!(
                "rir graph coverage: {:<24} {:<7} native retired - the generated variant is \
                 the only implementation",
                site.op, site.backend,
            );
        }
    }
    for site in &sites {
        // A device-class rejection is never a domain: it means the variant this
        // build published cannot run on this machine at all.
        let broken: Vec<&(String, u64)> = site
            .rejects
            .iter()
            .filter(|(r, _)| DEVICE_REJECTS.contains(&r.as_str()))
            .collect();
        assert!(
            broken.is_empty(),
            "{}/{}: non-contract fallback {broken:?} - the published variant does not run here",
            site.op,
            site.backend
        );
        let undeclared: Vec<&(String, u64)> = site
            .rejects
            .iter()
            .filter(|(r, _)| !site.domain.contains(r))
            .collect();
        assert!(
            undeclared.is_empty(),
            "{}/{}: {undeclared:?} fell back for a reason the registry does not declare \
             (domain={:?}); declare the restriction in rir_kernels::integrations, or lift it",
            site.op,
            site.backend,
            site.domain
        );
        // A pair whose native is retired has nothing to fall back to, so a
        // single native dispatch here is not a fallback - it is a count of a
        // kernel that does not exist. The runtime aborts before producing one,
        // which makes this assertion a check on the *report* as much as on the
        // run: the two must agree, or the coverage number published for a
        // retired pair means nothing.
        //
        // It is also the promotion criterion read on a real graph, not on a
        // conformance bench: on these pairs there is one implementation, and it
        // served every node the model produced.
        if site.retired {
            assert_eq!(
                site.native, 0,
                "{}/{}: {} native nodes on a pair whose native is retired",
                site.op, site.backend, site.native
            );
            assert!(
                site.domain.is_empty(),
                "{}/{}: native retired but a domain is declared ({:?}) - retiring the \
                 native kernel is only allowed when no restriction is published",
                site.op,
                site.backend,
                site.domain
            );
            assert_eq!(
                site.rir, site.seen,
                "{}/{}: {}/{} nodes served when there is no other path left",
                site.op, site.backend, site.rir, site.seen
            );
        }
        // Every node is accounted for: what RIR took plus what went native is
        // what the site saw. A gap would mean a node counted and then dropped.
        assert_eq!(
            site.rir + site.native,
            site.seen,
            "{}/{}: {} nodes seen but {} + {} accounted for",
            site.op,
            site.backend,
            site.seen,
            site.rir,
            site.native
        );
        // And every node that did **not** become eligible has a stated reason.
        //
        // Without this, the two checks above are satisfied by a path that
        // increments `native_dispatched` and never calls `count_reject`: the
        // reject list would be empty, so "every reject is declared" holds
        // vacuously, and `rir + native == seen` holds too. A silent fallback,
        // exactly what this file exists to catch - would pass green.
        //
        // Stated as `eligible + rejects == seen` rather than as the lane's
        // `native == rejects`, because that form is also true of an `observe`
        // pair: there a node can match the contract and still go native, so it
        // is eligible *and* counted native with no rejection to its name. The
        // equality below covers both policies without needing to know which one
        // is in force.
        let explained: u64 = site.rejects.iter().map(|(_, n)| n).sum();
        assert_eq!(
            site.eligible + explained,
            site.seen,
            "{}/{}: {} nodes seen, {} eligible, {explained} rejected with a reason - \
             {} fell back with no reason recorded",
            site.op,
            site.backend,
            site.seen,
            site.eligible,
            site.seen - site.eligible - explained
        );
    }

    // The backend that just dispatched nodes cannot be reported unavailable.
    // Checked on a site the graph actually produced rather than on one named
    // here: naming an op made this test a Qwen3.5 test, and it silently proved
    // nothing for every other model.
    let site = sites
        .iter()
        .find(|s| s.rir > 0)
        .expect("no site dispatched a single node to RIR under prefer");
    let variant_line = report
        .lines()
        .find(|l| {
            l.trim_start()
                .starts_with(&format!("{}/{}:", site.op, site.backend))
        })
        .unwrap_or_else(|| panic!("no variant row for {}/{}:\n{report}", site.op, site.backend));
    assert!(
        variant_line.ends_with(" available"),
        "{}/{} dispatched {} nodes but its variant is reported {variant_line:?}",
        site.op,
        site.backend,
        site.rir
    );
}

/// Rejections that are never a domain restriction: they say the variant this
/// build published cannot run on this machine, or that the selection chain is
/// not wired to the site at all. Same list as `scripts/test-rir.sh`.
const DEVICE_REJECTS: [&str; 6] = [
    "missing_feature",
    "pipeline",
    "device_grid",
    "device_alignment",
    "wrong_op",
    "policy_native",
];

/// One `(ggml_op, backend)` dispatch site, as the capability report publishes
/// it: the counters, the rejections that produced them, and the domain the
/// registry declared out of scope.
#[derive(Debug, Default)]
struct Site {
    op: String,
    backend: String,
    seen: u64,
    /// Nodes whose contract matched. Distinct from `rir`: an `observe` pair is
    /// eligible and still runs native, which is why the accounting identity is
    /// written against this rather than against `native`.
    eligible: u64,
    rir: u64,
    native: u64,
    rejects: Vec<(String, u64)>,
    domain: Vec<String>,
    /// Whether this pair's **native kernel no longer exists** in the build
    /// (`site-retired`). The strongest thing the
    /// report can say about a promotion, and the one thing `native=0` cannot:
    /// a lucky graph produces that on a pair whose native is very much there.
    retired: bool,
}

impl Site {
    fn reject_summary(&self) -> String {
        self.rejects
            .iter()
            .map(|(r, n)| format!("{r}={n}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The `site`, `site-reject` and `site-domain` rows of a capability report,
    /// joined on `(op, backend)`. Tab-separated and indented by the report's own
    /// section nesting, so every field is read after `trim_start`.
    fn parse(report: &str) -> Vec<Site> {
        let mut sites: Vec<Site> = Vec::new();
        let index = |sites: &[Site], op: &str, be: &str| {
            sites.iter().position(|s| s.op == op && s.backend == be)
        };
        for line in report.lines() {
            let f: Vec<&str> = line.trim_start().split('\t').collect();
            if f.len() < 3 {
                continue;
            }
            let (op, be) = (f[1].to_string(), f[2].to_string());
            match f[0] {
                "site" if f.len() >= 7 => {
                    let num = |s: &str| -> u64 {
                        s.split_once('=')
                            .and_then(|(_, v)| v.parse().ok())
                            .unwrap_or(0)
                    };
                    sites.push(Site {
                        op,
                        backend: be,
                        seen: num(f[3]),
                        eligible: num(f[4]),
                        rir: num(f[5]),
                        native: num(f[6]),
                        ..Site::default()
                    });
                }
                "site-reject" if f.len() >= 5 => {
                    if let Some(i) = index(&sites, &op, &be) {
                        sites[i]
                            .rejects
                            .push((f[3].to_string(), f[4].parse().unwrap_or(0)));
                    }
                }
                "site-domain" if f.len() >= 4 => {
                    if let Some(i) = index(&sites, &op, &be) {
                        sites[i].domain.push(f[3].to_string());
                    }
                }
                "site-retired" => {
                    if let Some(i) = index(&sites, &op, &be) {
                        sites[i].retired = true;
                    }
                }
                _ => {}
            }
        }
        sites
    }
}

/// The op census of the real backward graph.
///
/// The coverage test above asks whether the ops RIR *already* covers fall back.
/// This one asks the opposite question - which op is worth writing next - and
/// it needs the ops RIR does **not** cover, so it reads the census rather than
/// the site counters.
///
/// It asserts what a ranking has to be true of, not the ranking itself: an op
/// order is a property of the model, and pinning it here would make a different
/// model a test failure rather than a different answer. What is pinned is that
/// the census saw the graph, that it saw ops RIR does not cover (otherwise there
/// would be nothing left to write), and that its rows are internally
/// consistent. The ranking itself is printed.
#[test]
fn the_backward_graph_census_ranks_the_ops_worth_writing_next() {
    if std::env::var("RETRO_RIR_CENSUS").as_deref() != Ok("1") {
        eprintln!(
            "skipped: set RETRO_RIR_CENSUS=1 before the process starts - the census is \
             latched at the first graph, so a test cannot turn it on for itself"
        );
        return;
    }
    if !common::gpu_device_present() {
        eprintln!("skipped: no GPU device registered at runtime");
        return;
    }
    let Some(model) = model_path() else {
        eprintln!("skipped: set {MODEL_ENV} to the model whose graph should be censused");
        return;
    };
    assert!(
        model.exists(),
        "{MODEL_ENV} points at {} which does not exist",
        model.display()
    );

    let config = TrainConfig {
        n_ctx: 128,
        n_batch: 64,
        n_ubatch: 16,
        epochs: 1,
        learning_rate: 1.0e-4,
        device: Device::Gpu,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(&model, config).expect("load gpu trainer");
    let mut lora = LoraConfig::qv(1, 2.0);
    lora.seed = 7;
    lora.targets = TargetSet::Patterns(vec!["blk.*.attn_q.weight".to_string()]);
    trainer.create_lora(&lora).expect("create lora");
    let tokens = trainer
        .tokenize_text(&TRAIN_TEXT.repeat(12))
        .expect("tokenize");
    trainer.train_tokens(&tokens).expect("train one epoch");

    let report = rir_census_report().expect("rir_census_report");
    let mut rows: Vec<CensusRow> = report.lines().filter_map(CensusRow::parse).collect();
    assert!(
        !rows.is_empty(),
        "RETRO_RIR_CENSUS=1 and a GPU training step produced no census row:\n{report}"
    );

    rows.sort_by_key(|r| std::cmp::Reverse(r.bytes));
    let total_bytes: u64 = rows.iter().map(|r| r.bytes).sum();
    let total_nodes: u64 = rows.iter().map(|r| r.nodes).sum();
    eprintln!(
        "rir census: {total_nodes} nodes, {total_bytes} bytes, {} rows",
        rows.len()
    );
    for r in &rows {
        let share = 100.0 * r.bytes as f64 / total_bytes.max(1) as f64;
        eprintln!(
            "rir census: {:<28} {:<7} nodes={:<6} bytes={:<14} ({:5.2}%) {}",
            r.op, r.backend, r.nodes, r.bytes, share, r.coverage
        );
    }
    // The shape lines, which are what the isolated bench is fed next.
    for line in report.lines().filter(|l| l.starts_with("census-shape")) {
        eprintln!("rir census: {line}");
    }

    // A row with nodes but no bytes would mean the walk visited a node it could
    // not size, i.e. the ranking column is not the one the rows claim.
    for r in &rows {
        assert!(
            r.bytes > 0 && r.nodes > 0,
            "census row {r:?} carries a count without the traffic that ranks it"
        );
    }
    assert!(
        rows.iter().any(|r| r.coverage == "uncovered"),
        "every op of the graph is already registered - the census would have \
         nothing to rank:\n{report}"
    );

    // The chain rows. Same discipline as above: the
    // ranking is a property of the model and is printed, not asserted. What is
    // asserted is that the walk found chains at all - a training graph without
    // a single privately consumed intermediate would mean the fanout filter is
    // rejecting everything - and that each row's columns are consistent with
    // what a fusion of it would do.
    let patterns: Vec<CensusPattern> = report.lines().filter_map(CensusPattern::parse).collect();
    assert!(
        !patterns.is_empty(),
        "the census saw {total_nodes} nodes and no fusable chain at all:\n{report}"
    );
    let mut ranked = patterns.iter().collect::<Vec<_>>();
    ranked.sort_by_key(|p| std::cmp::Reverse((p.intermediate_bytes, p.dispatches)));
    eprintln!("rir census: {} chain patterns", ranked.len());
    for p in &ranked {
        eprintln!(
            "rir census: {:<44} {:<7} occ={:<6} saves {:<5} dispatches, {:<14} bytes round trip {}",
            p.ops,
            p.backend,
            p.occurrences,
            p.dispatches - p.occurrences,
            2 * p.intermediate_bytes,
            p.coverage
        );
    }
    for p in &ranked {
        assert!(
            p.ops.contains('>'),
            "a chain row of one op is not a fusion candidate: {p:?}"
        );
        let n_ops = p.ops.matches('>').count() as u64 + 1;
        assert!(
            (2..=4).contains(&n_ops),
            "chain row outside the published window length: {p:?}"
        );
        assert!(p.occurrences > 0, "chain row without an occurrence: {p:?}");
        assert_eq!(
            p.dispatches,
            p.occurrences * n_ops,
            "chain row whose dispatch count is not its length times its \
             occurrences, so the saving it publishes is not the one it \
             counted: {p:?}"
        );
    }
    // A shorter window can only be at least as frequent as the longer one it
    // sits inside: every occurrence of `a>b>c` is also an occurrence of `a>b`.
    // This is the one relation between rows the walk has to preserve, and the
    // one a fanout bug would break first.
    let table_full = report
        .lines()
        .any(|l| l.starts_with("census-pattern-overflow"));
    for long in &patterns {
        // Only from three ops up: the prefix of a two-op chain is one op, which
        // is not a fusion candidate and has no row by construction.
        if long.ops.matches('>').count() < 2 {
            continue;
        }
        let prefix = &long.ops[..long.ops.rfind('>').expect("a chain has a separator")];
        let Some(short) = patterns
            .iter()
            .find(|p| p.ops == prefix && p.backend == long.backend)
        else {
            assert!(
                table_full,
                "chain {} was counted but its prefix {prefix} was not, and no \
                 row was dropped",
                long.ops
            );
            continue;
        };
        assert!(
            short.occurrences >= long.occurrences,
            "prefix {prefix} occurs {} times but the longer chain {} occurs {} \
             times, which no walk over the same graph can produce",
            short.occurrences,
            long.ops,
            long.occurrences
        );
    }
}

#[derive(Debug)]
struct CensusPattern {
    ops: String,
    backend: String,
    occurrences: u64,
    dispatches: u64,
    intermediate_bytes: u64,
    coverage: String,
}

impl CensusPattern {
    fn parse(line: &str) -> Option<Self> {
        let mut f = line.split('\t');
        if f.next()? != "census-pattern" {
            return None;
        }
        Some(Self {
            ops: f.next()?.to_string(),
            backend: f.next()?.to_string(),
            occurrences: f.next()?.parse().ok()?,
            dispatches: f.next()?.parse().ok()?,
            intermediate_bytes: f.next()?.parse().ok()?,
            coverage: f.next()?.to_string(),
        })
    }
}

#[derive(Debug)]
struct CensusRow {
    op: String,
    backend: String,
    nodes: u64,
    bytes: u64,
    coverage: String,
}

impl CensusRow {
    fn parse(line: &str) -> Option<Self> {
        let mut f = line.split('\t');
        if f.next()? != "census" {
            return None;
        }
        Some(Self {
            op: f.next()?.to_string(),
            backend: f.next()?.to_string(),
            nodes: f.next()?.parse().ok()?,
            bytes: f.next()?.parse().ok()?,
            coverage: {
                let _elements = f.next()?;
                f.next()?.to_string()
            },
        })
    }
}
