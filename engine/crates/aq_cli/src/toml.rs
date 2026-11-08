//! A small TOML subset reader for `eval/suite.toml` (DESIGN 3.3 allows no toml crate).
//!
//! Supported: `# comments`, `[table]`, `[a.b]` (dotted table names), `[[array_of_tables]]`,
//! `key = value` with bare or quoted keys, basic strings `"..."` (escapes `\" \\ \n \t`),
//! literal strings `'...'`, integers, floats (including `inf`, `nan`), booleans, and
//! single-line arrays of those. The result is a `serde_json::Value` object.

use serde_json::{Map, Value};

/// Parses the text into a JSON object.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut root = Map::new();
    // Path of the current table: names, and whether the last one is an array element.
    let mut cur: Vec<String> = Vec::new();
    let mut cur_is_array = false;
    for (ln, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        let err = |m: &str| format!("line {}: {m}: {raw:?}", ln + 1);
        if let Some(inner) = line.strip_prefix("[[") {
            let name = inner
                .strip_suffix("]]")
                .ok_or_else(|| err("bad [[table]]"))?;
            cur = split_name(name).ok_or_else(|| err("bad table name"))?;
            cur_is_array = true;
            let (parent, last) = cur.split_at(cur.len() - 1);
            let t = table_mut(&mut root, parent, false).ok_or_else(|| err("table conflict"))?;
            let e = t
                .entry(last[0].clone())
                .or_insert_with(|| Value::Array(Vec::new()));
            match e {
                Value::Array(a) => a.push(Value::Object(Map::new())),
                _ => return Err(err("not an array of tables")),
            }
            continue;
        }
        if let Some(inner) = line.strip_prefix('[') {
            let name = inner.strip_suffix(']').ok_or_else(|| err("bad [table]"))?;
            cur = split_name(name).ok_or_else(|| err("bad table name"))?;
            cur_is_array = false;
            table_mut(&mut root, &cur, false).ok_or_else(|| err("table conflict"))?;
            continue;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| err("expected key = value"))?;
        let key = unquote_key(k.trim()).ok_or_else(|| err("bad key"))?;
        let val = parse_value(v.trim()).map_err(|m| err(&m))?;
        let t = table_mut(&mut root, &cur, cur_is_array).ok_or_else(|| err("table conflict"))?;
        if t.insert(key, val).is_some() {
            return Err(err("duplicate key"));
        }
    }
    Ok(Value::Object(root))
}

/// The line without its comment (a `#` outside a string).
fn strip_comment(s: &str) -> &str {
    let mut in_basic = false;
    let mut in_lit = false;
    let mut esc = false;
    for (i, c) in s.char_indices() {
        match c {
            _ if esc => esc = false,
            '\\' if in_basic => esc = true,
            '"' if !in_lit => in_basic = !in_basic,
            '\'' if !in_basic => in_lit = !in_lit,
            '#' if !in_basic && !in_lit => return &s[..i],
            _ => {}
        }
    }
    s
}

fn split_name(s: &str) -> Option<Vec<String>> {
    let parts: Vec<String> = s
        .split('.')
        .map(|p| unquote_key(p.trim()))
        .collect::<Option<_>>()?;
    if parts.is_empty() { None } else { Some(parts) }
}

fn unquote_key(k: &str) -> Option<String> {
    if k.len() >= 2
        && ((k.starts_with('"') && k.ends_with('"')) || (k.starts_with('\'') && k.ends_with('\'')))
    {
        return Some(k[1..k.len() - 1].to_string());
    }
    if !k.is_empty()
        && k.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        Some(k.to_string())
    } else {
        None
    }
}

/// The table at `path`, created when missing. When `last_is_array`, the last path element is
/// an array of tables and its last element is returned.
fn table_mut<'a>(
    root: &'a mut Map<String, Value>,
    path: &[String],
    last_is_array: bool,
) -> Option<&'a mut Map<String, Value>> {
    let mut t = root;
    for (i, name) in path.iter().enumerate() {
        let is_last = i + 1 == path.len();
        let e = t
            .entry(name.clone())
            .or_insert_with(|| Value::Object(Map::new()));
        t = match e {
            Value::Object(m) => m,
            Value::Array(a) if !is_last || last_is_array => match a.last_mut() {
                Some(Value::Object(m)) => m,
                _ => return None,
            },
            _ => return None,
        };
    }
    Some(t)
}

fn parse_value(s: &str) -> Result<Value, String> {
    if s.is_empty() {
        return Err("missing value".into());
    }
    if let Some(inner) = s.strip_prefix('[') {
        let inner = inner.strip_suffix(']').ok_or("unterminated array")?;
        let mut out = Vec::new();
        for item in split_array(inner)? {
            let item = item.trim();
            if !item.is_empty() {
                out.push(parse_value(item)?);
            }
        }
        return Ok(Value::Array(out));
    }
    if s.starts_with('"') {
        return parse_basic_string(s).map(Value::String);
    }
    if let Some(inner) = s.strip_prefix('\'') {
        let inner = inner.strip_suffix('\'').ok_or("unterminated string")?;
        return Ok(Value::String(inner.to_string()));
    }
    match s {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        "inf" | "+inf" => return Ok(Value::from(f64::MAX)),
        "-inf" => return Ok(Value::from(f64::MIN)),
        _ => {}
    }
    let clean: String = s.chars().filter(|&c| c != '_').collect();
    if let Ok(i) = clean.parse::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = clean.parse::<f64>()
        && f.is_finite()
    {
        return Ok(Value::from(f));
    }
    Err(format!("bad value {s:?}"))
}

/// Splits array items on top-level commas (strings may contain commas).
fn split_array(s: &str) -> Result<Vec<String>, String> {
    let mut items = Vec::new();
    let mut cur = String::new();
    let mut in_basic = false;
    let mut in_lit = false;
    let mut esc = false;
    for c in s.chars() {
        match c {
            _ if esc => {
                esc = false;
                cur.push(c);
                continue;
            }
            '\\' if in_basic => esc = true,
            '"' if !in_lit => in_basic = !in_basic,
            '\'' if !in_basic => in_lit = !in_lit,
            '[' | ']' if !in_basic && !in_lit => {
                return Err("nested arrays are not supported".into());
            }
            ',' if !in_basic && !in_lit => {
                items.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if in_basic || in_lit {
        return Err("unterminated string in array".into());
    }
    items.push(cur);
    Ok(items)
}

fn parse_basic_string(s: &str) -> Result<String, String> {
    let inner = s
        .strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .ok_or("unterminated string")?;
    let mut out = String::new();
    let mut it = inner.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(o) => return Err(format!("unsupported escape \\{o}")),
                None => return Err("dangling escape".into()),
            }
        } else if c == '"' {
            return Err("unescaped quote in string".into());
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_shape() {
        let t = r#"
# comment
[suite]
testdata = "../testdata"   # trailing comment
seeds = [0, 1, 2]
adaptive = 0.0

[gates.M1]
count_err = 0.10
big = 1_000

[[mesh]]
name = "torus"
target = 2000
loop_axis = 'all'
ok = true

[[mesh]]
name = "a # not a comment"
target = 5000
tags = ["x", "y, z"]
"#;
        let v = parse(t).unwrap();
        assert_eq!(v["suite"]["testdata"], "../testdata");
        assert_eq!(v["suite"]["seeds"], serde_json::json!([0, 1, 2]));
        assert_eq!(v["suite"]["adaptive"], 0.0);
        assert_eq!(v["gates"]["M1"]["count_err"], 0.10);
        assert_eq!(v["gates"]["M1"]["big"], 1000);
        let m = v["mesh"].as_array().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0]["loop_axis"], "all");
        assert_eq!(m[0]["ok"], true);
        assert_eq!(m[1]["name"], "a # not a comment");
        assert_eq!(m[1]["tags"], serde_json::json!(["x", "y, z"]));
    }

    #[test]
    fn errors() {
        assert!(parse("a = ").is_err());
        assert!(parse("a = 1\na = 2").is_err());
        assert!(parse("[x\n").is_err());
        assert!(parse("a = \"open").is_err());
        assert!(parse("a = [[1]]").is_err());
        assert!(parse("just words").is_err());
        assert!(parse("a = 1\n[a]").is_err());
    }
}
