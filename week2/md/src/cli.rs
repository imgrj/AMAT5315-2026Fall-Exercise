//! Dependency-light command-line parsing with the learning-sheet defaults.

use std::collections::HashMap;

use crate::ForceStrategy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegratorChoice {
    VelocityVerlet,
    Euler,
}

impl IntegratorChoice {
    pub fn name(self) -> &'static str {
        match self {
            Self::VelocityVerlet => "velocity-verlet",
            Self::Euler => "euler",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunCfg {
    pub force: ForceStrategy,
    pub integrator: IntegratorChoice,
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub ramp_to: Option<f64>,
    pub out: String,
}

impl Default for RunCfg {
    fn default() -> Self {
        Self {
            force: ForceStrategy::Cells,
            integrator: IntegratorChoice::VelocityVerlet,
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            ramp_to: None,
            out: "artifacts".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CheckCfg {
    pub input: String,
    pub drift_limit: f64,
    pub temperature_limit: f64,
    pub shape_limit: f64,
}

#[derive(Clone, Debug)]
pub struct VideoCfg {
    pub input: String,
    pub out: String,
    pub fps: f64,
}

#[derive(Clone, Debug)]
pub enum Command {
    Run(RunCfg),
    Check(CheckCfg),
    Video(VideoCfg),
}

pub fn parse(args: &[String]) -> Result<Command, String> {
    let subcommand = args.first().ok_or("missing subcommand")?;
    let (flags, positional) = parse_flags(&args[1..])?;
    match subcommand.as_str() {
        "run" => parse_run(flags, positional),
        "check" => parse_check(flags, positional),
        "video" => parse_video(flags, positional),
        other => Err(format!("unknown subcommand {other:?}")),
    }
}

fn parse_flags(args: &[String]) -> Result<(HashMap<String, String>, Vec<String>), String> {
    let mut flags = HashMap::new();
    let mut positional = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index].starts_with('-') {
            let key = canonical(args[index].trim_start_matches('-'));
            let value = args.get(index + 1).ok_or_else(|| format!("--{key} needs a value"))?;
            if value.starts_with('-') {
                return Err(format!("--{key} needs a value"));
            }
            if flags.insert(key.to_owned(), value.clone()).is_some() {
                return Err(format!("--{key} was supplied more than once"));
            }
            index += 2;
        } else {
            positional.push(args[index].clone());
            index += 1;
        }
    }
    Ok((flags, positional))
}

fn canonical(name: &str) -> &str {
    match name {
        "equil" => "eq-steps",
        "temp" => "temperature",
        other => other,
    }
}

fn reject_unknown(flags: &HashMap<String, String>, allowed: &[&str]) -> Result<(), String> {
    if let Some(key) = flags.keys().find(|key| !allowed.contains(&key.as_str())) {
        Err(format!("unknown flag --{key}"))
    } else {
        Ok(())
    }
}

fn number<T: std::str::FromStr>(
    flags: &HashMap<String, String>,
    name: &str,
) -> Result<Option<T>, String> {
    flags
        .get(name)
        .map(|value| value.parse().map_err(|_| format!("--{name} has invalid value {value:?}")))
        .transpose()
}

fn parse_run(flags: HashMap<String, String>, positional: Vec<String>) -> Result<Command, String> {
    if !positional.is_empty() {
        return Err("run does not accept positional arguments".into());
    }
    reject_unknown(
        &flags,
        &[
            "force",
            "integrator",
            "n",
            "rho",
            "temperature",
            "dt",
            "eq-steps",
            "steps",
            "sample-every",
            "seed",
            "ramp-to",
            "out",
        ],
    )?;
    let mut cfg = RunCfg::default();
    cfg.n = number(&flags, "n")?.unwrap_or(cfg.n);
    cfg.rho = number(&flags, "rho")?.unwrap_or(cfg.rho);
    cfg.temperature = number(&flags, "temperature")?.unwrap_or(cfg.temperature);
    cfg.dt = number(&flags, "dt")?.unwrap_or(cfg.dt);
    cfg.eq_steps = number(&flags, "eq-steps")?.unwrap_or(cfg.eq_steps);
    cfg.steps = number(&flags, "steps")?.unwrap_or(cfg.steps);
    cfg.sample_every = number(&flags, "sample-every")?.unwrap_or(cfg.sample_every);
    cfg.seed = number(&flags, "seed")?.unwrap_or(cfg.seed);
    cfg.ramp_to = number(&flags, "ramp-to")?;
    cfg.out = flags.get("out").cloned().unwrap_or(cfg.out);
    cfg.force = match flags.get("force").map(String::as_str) {
        None | Some("cells") => ForceStrategy::Cells,
        Some("naive") => ForceStrategy::Naive,
        Some(value) => return Err(format!("--force must be naive or cells, got {value:?}")),
    };
    cfg.integrator = match flags.get("integrator").map(String::as_str) {
        None | Some("velocity-verlet") | Some("verlet") => IntegratorChoice::VelocityVerlet,
        Some("euler") => IntegratorChoice::Euler,
        Some(value) => return Err(format!("--integrator must be velocity-verlet or euler, got {value:?}")),
    };
    validate_run(&cfg)?;
    Ok(Command::Run(cfg))
}

fn validate_run(cfg: &RunCfg) -> Result<(), String> {
    if cfg.n == 0 || cfg.steps == 0 || cfg.sample_every == 0 || cfg.sample_every > cfg.steps {
        return Err("--n, --steps, and --sample-every must describe a nonempty run".into());
    }
    for (name, value) in [
        ("rho", cfg.rho),
        ("temperature", cfg.temperature),
        ("dt", cfg.dt),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("--{name} must be finite and positive"));
        }
    }
    if cfg.ramp_to.is_some_and(|x| !x.is_finite() || x <= 0.0) {
        return Err("--ramp-to must be finite and positive".into());
    }
    Ok(())
}

fn parse_check(flags: HashMap<String, String>, positional: Vec<String>) -> Result<Command, String> {
    reject_unknown(&flags, &["drift-limit", "temperature-limit", "shape-limit"])?;
    if positional.len() != 1 {
        return Err("check needs exactly one artifact directory".into());
    }
    Ok(Command::Check(CheckCfg {
        input: positional[0].clone(),
        drift_limit: number(&flags, "drift-limit")?.unwrap_or(2e-3),
        temperature_limit: number(&flags, "temperature-limit")?.unwrap_or(0.05),
        shape_limit: number(&flags, "shape-limit")?.unwrap_or(2.0),
    }))
}

fn parse_video(flags: HashMap<String, String>, positional: Vec<String>) -> Result<Command, String> {
    reject_unknown(&flags, &["out", "fps"])?;
    if positional.len() != 1 {
        return Err("video needs exactly one artifact directory".into());
    }
    let fps: f64 = number(&flags, "fps")?.unwrap_or(20.0);
    if !fps.is_finite() || fps <= 0.0 {
        return Err("--fps must be finite and positive".into());
    }
    Ok(Command::Video(VideoCfg {
        input: positional[0].clone(),
        out: flags.get("out").cloned().unwrap_or_else(|| "run.mp4".into()),
        fps,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn run_defaults_are_the_contract() {
        let Command::Run(cfg) = parse(&strings(&["run"])).unwrap() else { panic!() };
        assert_eq!(cfg.n, 100);
        assert_eq!(cfg.rho, 0.8);
        assert_eq!(cfg.temperature, 0.5);
        assert_eq!(cfg.dt, 0.01);
        assert_eq!(cfg.eq_steps, 2000);
        assert_eq!(cfg.steps, 10000);
        assert_eq!(cfg.sample_every, 50);
        assert_eq!(cfg.seed, 2026);
        assert_eq!(cfg.out, "artifacts");
        assert_eq!(cfg.force, ForceStrategy::Cells);
    }

    #[test]
    fn flags_override_defaults_and_aliases_work() {
        let Command::Run(cfg) = parse(&strings(&[
            "run", "--n", "400", "--rho", "0.7", "--temp", "0.2",
            "--equil", "10", "--steps", "20", "--sample-every", "2",
            "--force", "naive", "--integrator", "euler", "--ramp-to", "1.2",
            "--out", "x", "--seed", "9",
        ])).unwrap() else { panic!() };
        assert_eq!(cfg.n, 400);
        assert_eq!(cfg.eq_steps, 10);
        assert_eq!(cfg.force, ForceStrategy::Naive);
        assert_eq!(cfg.integrator, IntegratorChoice::Euler);
        assert_eq!(cfg.ramp_to, Some(1.2));
    }

    #[test]
    fn bad_or_unknown_flags_are_rejected() {
        assert!(parse(&strings(&["run", "--steps", "0"])).is_err());
        assert!(parse(&strings(&["run", "--force", "magic"])).is_err());
        assert!(parse(&strings(&["run", "--mystery", "1"])).is_err());
        assert!(parse(&strings(&["check"])).is_err());
        assert!(parse(&strings(&["video", "a", "b"])).is_err());
    }
}
