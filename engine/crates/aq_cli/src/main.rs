//! # autoquad.exe: the AutoQuad engine CLI (DESIGN 9, WP-05)
//!
//! ```text
//! autoquad remesh <in.obj> -o <out.obj> [--target N] [--adaptive F] [--align F] [--smooth F]
//!         [--features auto|none|<file>] [--hard-angle DEG(30)] [--density <file>] [--sym x|y|z] [--keep-neg]
//!         [--seed N] [--threads N] [--time-limit S] [--no-loops] [--no-collapse] [--upsample K]
//!         [--reproject] [--report <r.json>] [--dump <dir>] [--stop-after S0..S11] [--axis x,y,z]
//! autoquad batch --suite eval/suite.toml --out <dir> [--seeds 0..4] [--compare baselines/] [--markdown summary.md]
//!         [--gate M0|M1|M2] [--only name,name]
//! autoquad stats <quad.obj> [--ref <input.obj>] [--axis x,y,z] [--target N] [--json]
//! ```
//!
//! Exit codes: 0 success (status 0 or 1), 2 engine status 2, 3 cancelled, 4 invalid input,
//! 5 caught panic, 1 file or other error, 64 usage error. `batch` exits 0 when every run
//! executed (gate failures are reported, not fatal) unless `--strict` is given (then 6 when a
//! mesh fails its gates).

mod args;
mod batch;
mod flow;
mod stats;
mod toml;

use std::path::PathBuf;
use std::process::ExitCode;

use args::{Args, Spec, parse_axis, parse_seeds, parse_stage};
use flow::{Features, RemeshOpts};

const USAGE: &str = "usage:
  autoquad remesh <in.obj> -o <out.obj> [--target N] [--adaptive F] [--align F] [--smooth F]
          [--features auto|none|<file>] [--hard-angle DEG] [--density <file>] [--sym x|y|z] [--keep-neg]
          [--seed N] [--threads N] [--time-limit S] [--no-loops] [--no-collapse] [--upsample K]
          [--reproject] [--report <r.json>] [--dump <dir>] [--stop-after S0..S11] [--axis x,y,z]
  autoquad batch --suite eval/suite.toml --out <dir> [--seeds 0..4] [--compare baselines/]
          [--markdown summary.md] [--gate M0|M1|M2] [--only a,b] [--strict]
  autoquad stats <quad.obj> [--ref <input.obj>] [--axis x,y,z] [--target N] [--json]
  autoquad version";

const REMESH_SPEC: Spec = &[
    ("o", true),
    ("target", true),
    ("adaptive", true),
    ("align", true),
    ("smooth", true),
    ("features", true),
    ("hard-angle", true),
    ("density", true),
    ("sym", true),
    ("keep-neg", false),
    ("seed", true),
    ("threads", true),
    ("time-limit", true),
    ("no-loops", false),
    ("no-collapse", false),
    ("upsample", true),
    ("reproject", false),
    ("report", true),
    ("dump", true),
    ("stop-after", true),
    ("axis", true),
];

const BATCH_SPEC: Spec = &[
    ("suite", true),
    ("out", true),
    ("seeds", true),
    ("compare", true),
    ("markdown", true),
    ("gate", true),
    ("only", true),
    ("strict", false),
];

const STATS_SPEC: Spec = &[
    ("ref", true),
    ("axis", true),
    ("json", false),
    ("target", true),
];

enum Fail {
    Usage(String),
    Other(String),
}

impl From<String> for Fail {
    fn from(s: String) -> Fail {
        Fail::Other(s)
    }
}

fn usage(s: impl Into<String>) -> Fail {
    Fail::Usage(s.into())
}

/// Builds the remesh options from the arguments.
fn remesh_opts(a: &Args) -> Result<RemeshOpts, Fail> {
    let [input] = a.positional.as_slice() else {
        return Err(usage("remesh takes exactly one input OBJ"));
    };
    let mut o = RemeshOpts::new(PathBuf::from(input));
    o.output = a.get("o").map(PathBuf::from);
    if o.output.is_none() {
        return Err(usage("remesh needs -o <out.obj>"));
    }
    let p = &mut o.params;
    let u = |e: String| Fail::Usage(e);
    if let Some(x) = a.parse_opt::<u32>("target").map_err(u)? {
        p.target_quads = x;
    }
    if let Some(x) = a.parse_opt::<f32>("adaptive").map_err(u)? {
        p.adaptive_size = x;
    }
    if let Some(x) = a.parse_opt::<f32>("align").map_err(u)? {
        p.curvature_align = x;
    }
    if let Some(x) = a.parse_opt::<f32>("smooth").map_err(u)? {
        p.flow_smoothness = x;
    }
    if let Some(x) = a.parse_opt::<u64>("seed").map_err(u)? {
        p.seed = x;
    }
    if let Some(x) = a.parse_opt::<u32>("threads").map_err(u)? {
        p.threads = x;
    }
    if let Some(x) = a.parse_opt::<f32>("time-limit").map_err(u)? {
        p.time_limit_s = x;
    }
    if a.flag("no-loops") {
        p.close_loops = 0;
    }
    if a.flag("no-collapse") {
        p.allow_collapse = 0;
    }
    if let Some(d) = a.get("dump") {
        p.dump_dir = Some(PathBuf::from(d));
        p.debug_flags |= aq_engine::debug::DUMP;
    }
    if let Some(s) = a.get("stop-after") {
        p.stop_after = Some(parse_stage(s).map_err(u)?);
    }
    o.features = match a.get("features") {
        None | Some("none") => Features::None,
        Some("auto") => {
            o.params.hard_edges = 2;
            Features::Auto
        }
        Some(f) => Features::File(PathBuf::from(f)),
    };
    if let Some(x) = a.parse_opt::<f64>("hard-angle").map_err(u)? {
        if !(x.is_finite() && x > 0.0 && x < 180.0) {
            return Err(usage(format!("--hard-angle {x} must be in (0, 180)")));
        }
        o.hard_angle = x;
    }
    o.density = a.get("density").map(PathBuf::from);
    if let Some(s) = a.get("sym") {
        let mut axes = Vec::new();
        for c in s.chars() {
            let i = "xyz"
                .find(c.to_ascii_lowercase())
                .ok_or_else(|| usage(format!("--sym {s}: axes are x, y, z")))?;
            if !axes.contains(&i) {
                axes.push(i);
            }
        }
        if axes.is_empty() {
            return Err(usage("--sym needs x, y or z"));
        }
        o.sym = axes;
    }
    o.keep_neg = a.flag("keep-neg");
    if let Some(k) = a.parse_opt::<u32>("upsample").map_err(u)? {
        if k > 4 {
            return Err(usage("--upsample K: K is at most 4"));
        }
        o.upsample = k;
    }
    o.reproject = a.flag("reproject");
    o.report = a.get("report").map(PathBuf::from);
    if let Some(ax) = a.get("axis") {
        o.axis = Some(parse_axis(ax).map_err(u)?);
    }
    Ok(o)
}

fn cmd_remesh(argv: &[String]) -> Result<ExitCode, Fail> {
    let a = Args::parse(argv, REMESH_SPEC).map_err(Fail::Usage)?;
    let o = remesh_opts(&a)?;
    let oc = flow::run(&o)?;
    eprintln!(
        "status {} ({}): {} faces for target {} in {:.2} s{}",
        oc.status.code(),
        oc.status.name(),
        oc.mesh.n_faces(),
        o.params.target_quads,
        oc.seconds,
        if oc.error.is_empty() {
            String::new()
        } else {
            format!(", {}", oc.error)
        }
    );
    Ok(ExitCode::from(match oc.status.code() {
        0 | 1 => 0,
        c => c as u8,
    }))
}

fn cmd_batch(argv: &[String]) -> Result<ExitCode, Fail> {
    let a = Args::parse(argv, BATCH_SPEC).map_err(Fail::Usage)?;
    if !a.positional.is_empty() {
        return Err(usage("batch takes no positional argument"));
    }
    let suite = a.get("suite").ok_or_else(|| usage("batch needs --suite"))?;
    let out = a.get("out").ok_or_else(|| usage("batch needs --out"))?;
    let seeds = match a.get("seeds") {
        Some(s) => Some(parse_seeds(s).map_err(Fail::Usage)?),
        None => None,
    };
    let milestone = a.get("gate").unwrap_or("M1").to_ascii_uppercase();
    let only = a.get("only").map_or(Vec::new(), |s| {
        s.split(',').map(|x| x.trim().to_string()).collect()
    });
    let o = batch::BatchOpts {
        suite: PathBuf::from(suite),
        out: PathBuf::from(out),
        seeds,
        compare: a.get("compare").map(PathBuf::from),
        markdown: a.get("markdown").map(PathBuf::from),
        milestone,
        only,
    };
    let r = batch::run_batch(&o)?;
    let n_ok = r.verdicts.iter().filter(|v| v.4).count();
    eprintln!(
        "{n_ok} of {} meshes pass the {} gates; summary in {}",
        r.verdicts.len(),
        o.milestone,
        o.out.join("summary.json").display()
    );
    if a.flag("strict") && n_ok < r.verdicts.len() {
        return Ok(ExitCode::from(6));
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_stats(argv: &[String]) -> Result<ExitCode, Fail> {
    let a = Args::parse(argv, STATS_SPEC).map_err(Fail::Usage)?;
    let [path] = a.positional.as_slice() else {
        return Err(usage("stats takes exactly one quad OBJ"));
    };
    let q = aq_io::read_obj_poly(&PathBuf::from(path)).map_err(|e| format!("{path}: {e}"))?;
    let input = match a.get("ref") {
        Some(r) => Some(aq_io::read_obj_tri(&PathBuf::from(r)).map_err(|e| format!("{r}: {e}"))?),
        None => None,
    };
    let axis = match a.get("axis") {
        Some(x) => Some(parse_axis(x).map_err(Fail::Usage)?),
        None => None,
    };
    let target = a.parse_opt::<u32>("target").map_err(Fail::Usage)?;
    let s = stats::compute(&q, input.as_ref(), target, axis, None);
    if a.flag("json") {
        println!(
            "{}",
            serde_json::to_string_pretty(&s.to_json()).unwrap_or_default()
        );
    } else {
        print!("{}", s.to_text());
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(64);
    };
    let rest = &argv[1..];
    let r = match cmd.as_str() {
        "remesh" => cmd_remesh(rest),
        "batch" => cmd_batch(rest),
        "stats" => cmd_stats(rest),
        "version" | "--version" => {
            println!(
                "autoquad {} (engine {}, ABI {})",
                env!("CARGO_PKG_VERSION"),
                aq_engine::ENGINE_PATH,
                aq_engine::ABI_VERSION
            );
            Ok(ExitCode::SUCCESS)
        }
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        other => Err(Fail::Usage(format!("unknown command {other:?}"))),
    };
    match r {
        Ok(c) => c,
        Err(Fail::Usage(e)) => {
            eprintln!("error: {e}\n{USAGE}");
            ExitCode::from(64)
        }
        Err(Fail::Other(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
