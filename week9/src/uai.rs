//! MARKOV UAI files, evidence files and the published PR/MAR answer files.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// One conditional probability (or potential) table with its declared axis order.
#[derive(Clone, Debug)]
pub struct Factor {
    /// Variable IDs in table axis order; the last axis varies fastest in the file.
    pub scope: Vec<usize>,
    pub dims: Vec<usize>,
    /// `dims.iter().product()` entries in scope order.
    pub entries: Vec<f64>,
}

/// A parsed `MARKOV` model.
#[derive(Clone, Debug)]
pub struct Model {
    pub path: PathBuf,
    pub variables: usize,
    pub dims: Vec<usize>,
    pub factors: Vec<Factor>,
}

/// Observations as `(variable, state)` pairs in file order.
pub type Observations = Vec<(usize, usize)>;

#[derive(Debug)]
pub struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

fn tokens(path: &Path) -> Result<Vec<String>, ParseError> {
    let text = fs::read_to_string(path)
        .map_err(|error| ParseError(format!("cannot read {}: {error}", path.display())))?;
    Ok(text.split_whitespace().map(str::to_owned).collect())
}

fn number<T: std::str::FromStr>(token: &str, what: &str) -> Result<T, ParseError> {
    token
        .parse::<T>()
        .map_err(|_| ParseError(format!("expected {what}, found {token:?}")))
}

impl Model {
    pub fn read(path: &Path) -> Result<Self, ParseError> {
        let tokens = tokens(path)?;
        let mut at = 0usize;
        let mut take = |what: &str| -> Result<String, ParseError> {
            let token = tokens
                .get(at)
                .cloned()
                .ok_or_else(|| ParseError(format!("truncated file: missing {what}")))?;
            at += 1;
            Ok(token)
        };
        if take("format line")? != "MARKOV" {
            return Err(ParseError("only MARKOV UAI files are supported".into()));
        }
        let variables: usize = number(&take("variable count")?, "variable count")?;
        let mut dims = Vec::with_capacity(variables);
        for _ in 0..variables {
            dims.push(number(&take("variable dimension")?, "variable dimension")?);
        }
        let count: usize = number(&take("factor count")?, "factor count")?;
        let mut scopes = Vec::with_capacity(count);
        for _ in 0..count {
            let rank: usize = number(&take("scope size")?, "scope size")?;
            let mut scope = Vec::with_capacity(rank);
            for _ in 0..rank {
                let variable: usize = number(&take("scope variable")?, "scope variable")?;
                if variable >= variables {
                    return Err(ParseError(format!(
                        "scope variable {variable} outside 0..{variables}"
                    )));
                }
                if scope.contains(&variable) {
                    return Err(ParseError(format!("repeated variable {variable} in a scope")));
                }
                scope.push(variable);
            }
            scopes.push(scope);
        }
        let mut factors = Vec::with_capacity(count);
        for scope in scopes {
            let dims_of: Vec<usize> = scope.iter().map(|variable| dims[*variable]).collect();
            let expected: usize = dims_of.iter().product();
            let size: usize = number(&take("table size")?, "table size")?;
            if size != expected {
                return Err(ParseError(format!(
                    "scope {scope:?} needs {expected} entries, file says {size}"
                )));
            }
            let mut entries = Vec::with_capacity(size);
            for _ in 0..size {
                entries.push(number(&take("table entry")?, "table entry")?);
            }
            factors.push(Factor {
                scope,
                dims: dims_of,
                entries,
            });
        }
        if at != tokens.len() {
            return Err(ParseError(format!(
                "{} trailing tokens after the last table",
                tokens.len() - at
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            variables,
            dims,
            factors,
        })
    }

    /// A file next to the model, when present: `<model><suffix>`.
    pub fn sibling(&self, suffix: &str) -> Option<PathBuf> {
        let mut name = self.path.file_name()?.to_os_string();
        name.push(suffix);
        let candidate = self.path.with_file_name(name);
        candidate.exists().then_some(candidate)
    }

    /// Evidence file next to the model, when present: `<model>.evid`.
    pub fn sibling_evidence(&self) -> Option<PathBuf> {
        self.sibling(".evid")
    }

    pub fn sha256(&self) -> String {
        sha256_file(&self.path)
    }
}

pub fn read_evidence(path: &Path) -> Result<Observations, ParseError> {
    let tokens = tokens(path)?;
    let count: usize = number(
        tokens.first().map(String::as_str).unwrap_or(""),
        "observation count",
    )?;
    if tokens.len() != 1 + 2 * count {
        return Err(ParseError(format!(
            "{}: expected {count} observations, found {} tokens",
            path.display(),
            tokens.len() - 1
        )));
    }
    Ok((0..count)
        .map(|index| {
            let variable = number::<usize>(&tokens[1 + 2 * index], "variable").unwrap_or(usize::MAX);
            let state = number::<usize>(&tokens[2 + 2 * index], "state").unwrap_or(usize::MAX);
            (variable, state)
        })
        .collect())
}

/// State-ordered marginals from a `.uai.MAR` answer file.
pub fn read_mar(path: &Path) -> Result<Vec<Vec<f64>>, ParseError> {
    let tokens = tokens(path)?;
    if tokens.first().map(String::as_str) != Some("MAR") {
        return Err(ParseError("not a MAR file".into()));
    }
    let count: usize = number(&tokens[1], "variable count")?;
    let mut at = 2usize;
    let mut marginals = Vec::with_capacity(count);
    for _ in 0..count {
        let size: usize = number(&tokens[at], "marginal size")?;
        at += 1;
        if at + size > tokens.len() {
            return Err(ParseError("truncated MAR file".into()));
        }
        marginals.push(
            tokens[at..at + size]
                .iter()
                .map(|token| number::<f64>(token, "marginal").unwrap_or(f64::NAN))
                .collect(),
        );
        at += size;
    }
    Ok(marginals)
}

/// Published `log10 Z` from a `.uai.PR` answer file.
pub fn read_pr(path: &Path) -> Result<f64, ParseError> {
    let tokens = tokens(path)?;
    if tokens.first().map(String::as_str) != Some("PR") {
        return Err(ParseError("not a PR file".into()));
    }
    number(&tokens[1], "log10 partition sum")
}

pub fn sha256_file(path: &Path) -> String {
    match fs::read(path) {
        Ok(bytes) => {
            let digest = Sha256::digest(&bytes);
            digest.iter().map(|byte| format!("{byte:02x}")).collect()
        }
        Err(_) => String::new(),
    }
}
