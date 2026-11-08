//! Hand-written argument parsing (DESIGN 3.3: no clap).
//!
//! Arguments are split into positionals and `--name value` / `--name` options; a short
//! option such as `-o <path>` is the option named "o", and `--name=value` is also accepted.
//! Unknown options are an error, reported with the command's usage.

use std::collections::BTreeMap;

/// Parsed arguments of one command.
#[derive(Debug, Default)]
pub struct Args {
    pub positional: Vec<String>,
    opts: BTreeMap<String, Option<String>>,
}

/// Option spec: name and whether it takes a value.
pub type Spec = &'static [(&'static str, bool)];

impl Args {
    /// Parses `argv` against the spec. `-o` is read as option `o`.
    pub fn parse(argv: &[String], spec: Spec) -> Result<Args, String> {
        let mut a = Args::default();
        let mut i = 0;
        while i < argv.len() {
            let s = &argv[i];
            let (name, inline): (String, Option<String>) = if let Some(rest) = s.strip_prefix("--")
            {
                match rest.split_once('=') {
                    Some((n, v)) => (n.to_string(), Some(v.to_string())),
                    None => (rest.to_string(), None),
                }
            } else if s.len() == 2
                && s.starts_with('-')
                && !s[1..].starts_with(|c: char| c.is_ascii_digit())
            {
                (s[1..].to_string(), None)
            } else {
                a.positional.push(s.clone());
                i += 1;
                continue;
            };
            let Some(&(_, takes)) = spec.iter().find(|(n, _)| *n == name) else {
                return Err(format!("unknown option {s}"));
            };
            let val = if takes {
                match inline {
                    Some(v) => Some(v),
                    None => {
                        i += 1;
                        Some(
                            argv.get(i)
                                .cloned()
                                .ok_or_else(|| format!("option {s} needs a value"))?,
                        )
                    }
                }
            } else {
                if inline.is_some() {
                    return Err(format!("option --{name} takes no value"));
                }
                None
            };
            if a.opts.insert(name.clone(), val).is_some() {
                return Err(format!("option {s} given twice"));
            }
            i += 1;
        }
        Ok(a)
    }

    pub fn flag(&self, name: &str) -> bool {
        self.opts.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.opts.get(name).and_then(|v| v.as_deref())
    }

    /// A typed option value, `Ok(None)` when absent.
    pub fn parse_opt<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        match self.get(name) {
            None => Ok(None),
            Some(v) => v
                .parse::<T>()
                .map(Some)
                .map_err(|_| format!("bad value for --{name}: {v:?}")),
        }
    }
}

/// Parses a seed list: `0..4` (inclusive), `0,2,3`, or `7`.
pub fn parse_seeds(s: &str) -> Result<Vec<u64>, String> {
    let s = s.trim();
    if let Some((a, b)) = s.split_once("..") {
        let a: u64 = a
            .trim()
            .parse()
            .map_err(|_| format!("bad seed range {s:?}"))?;
        let b: u64 = b
            .trim()
            .parse()
            .map_err(|_| format!("bad seed range {s:?}"))?;
        if b < a || b - a > 1000 {
            return Err(format!("bad seed range {s:?}"));
        }
        return Ok((a..=b).collect());
    }
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<u64>()
                .map_err(|_| format!("bad seed {x:?}"))
        })
        .collect()
}

/// Parses an axis: `x`, `y`, `z`, or three comma-separated numbers.
pub fn parse_axis(s: &str) -> Result<[f64; 3], String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "x" => return Ok([1.0, 0.0, 0.0]),
        "y" => return Ok([0.0, 1.0, 0.0]),
        "z" => return Ok([0.0, 0.0, 1.0]),
        _ => {}
    }
    let v: Vec<f64> = s
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<f64>()
                .map_err(|_| format!("bad axis {s:?}"))
        })
        .collect::<Result<_, _>>()?;
    if v.len() != 3 || !v.iter().all(|x| x.is_finite()) || v.iter().all(|&x| x == 0.0) {
        return Err(format!("bad axis {s:?} (x, y, z or three numbers)"));
    }
    Ok([v[0], v[1], v[2]])
}

/// Parses `S0`..`S11` (or a bare number) into the stage number.
pub fn parse_stage(s: &str) -> Result<u8, String> {
    let t = s.trim().trim_start_matches(['S', 's']);
    let n: u8 = t
        .parse()
        .map_err(|_| format!("bad stage {s:?} (S0..S11)"))?;
    if n > 11 {
        return Err(format!("bad stage {s:?} (S0..S11)"));
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    const SPEC: Spec = &[
        ("o", true),
        ("target", true),
        ("reproject", false),
        ("seed", true),
    ];

    #[test]
    fn options_and_positionals() {
        let a = Args::parse(
            &v(&[
                "in.obj",
                "-o",
                "out.obj",
                "--target",
                "300",
                "--reproject",
                "--seed=4",
            ]),
            SPEC,
        )
        .unwrap();
        assert_eq!(a.positional, vec!["in.obj"]);
        assert_eq!(a.get("o"), Some("out.obj"));
        assert_eq!(a.parse_opt::<u32>("target").unwrap(), Some(300));
        assert!(a.flag("reproject"));
        assert_eq!(a.parse_opt::<u64>("seed").unwrap(), Some(4));
        assert!(!a.flag("seed2"));
    }

    #[test]
    fn errors() {
        assert!(Args::parse(&v(&["--nope"]), SPEC).is_err());
        assert!(Args::parse(&v(&["--target"]), SPEC).is_err());
        assert!(Args::parse(&v(&["--target", "1", "--target", "2"]), SPEC).is_err());
        assert!(Args::parse(&v(&["--reproject=1"]), SPEC).is_err());
        let a = Args::parse(&v(&["--target", "x"]), SPEC).unwrap();
        assert!(a.parse_opt::<u32>("target").is_err());
        // A negative number is a positional, not an option.
        let a = Args::parse(&v(&["-5"]), SPEC).unwrap();
        assert_eq!(a.positional, vec!["-5"]);
    }

    #[test]
    fn seeds_axes_stages() {
        assert_eq!(parse_seeds("0..4").unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(parse_seeds("1,3").unwrap(), vec![1, 3]);
        assert_eq!(parse_seeds("7").unwrap(), vec![7]);
        assert!(parse_seeds("4..1").is_err());
        assert_eq!(parse_axis("Z").unwrap(), [0.0, 0.0, 1.0]);
        assert_eq!(parse_axis("1,2,3").unwrap(), [1.0, 2.0, 3.0]);
        assert!(parse_axis("0,0,0").is_err());
        assert!(parse_axis("1,2").is_err());
        assert_eq!(parse_stage("S11").unwrap(), 11);
        assert_eq!(parse_stage("s0").unwrap(), 0);
        assert!(parse_stage("S12").is_err());
    }
}
