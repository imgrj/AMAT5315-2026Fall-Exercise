//! Strict JSON metadata and JSON Lines trajectory IO using only the standard library.

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::Path;

use crate::ForceStrategy;

pub const RUN_FILE: &str = "run.json";
pub const TRAJECTORY_FILE: &str = "traj.jsonl";

#[derive(Clone, Debug)]
pub struct RunMetadata {
    pub n: usize,
    pub rho: f64,
    pub box_size: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    pub force: ForceStrategy,
    pub ramp_to: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub e_pot: f64,
    pub e_kin: f64,
}

#[derive(Clone, Debug)]
pub struct Trajectory {
    pub meta: RunMetadata,
    pub frames: Vec<Frame>,
}

pub fn write(out: &Path, trajectory: &Trajectory) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let run_path = out.join(RUN_FILE);
    let mut run = BufWriter::new(
        File::create(&run_path).map_err(|e| format!("create {}: {e}", run_path.display()))?,
    );
    let ramp = trajectory
        .meta
        .ramp_to
        .map(|x| format!(",\n  \"ramp_to\": {x}"))
        .unwrap_or_default();
    writeln!(
        run,
        "{{\n  \"n\": {},\n  \"rho\": {},\n  \"box\": [{}, {}],\n  \"dt\": {},\n  \"temperature\": {},\n  \"eq_steps\": {},\n  \"steps\": {},\n  \"sample_every\": {},\n  \"seed\": {},\n  \"integrator\": \"{}\",\n  \"force\": \"{}\"{}\n}}",
        trajectory.meta.n,
        trajectory.meta.rho,
        trajectory.meta.box_size[0],
        trajectory.meta.box_size[1],
        trajectory.meta.dt,
        trajectory.meta.temperature,
        trajectory.meta.eq_steps,
        trajectory.meta.steps,
        trajectory.meta.sample_every,
        trajectory.meta.seed,
        trajectory.meta.integrator,
        trajectory.meta.force.as_str(),
        ramp
    )
    .map_err(|e| format!("write {}: {e}", run_path.display()))?;

    let traj_path = out.join(TRAJECTORY_FILE);
    let mut output = BufWriter::new(
        File::create(&traj_path).map_err(|e| format!("create {}: {e}", traj_path.display()))?,
    );
    for frame in &trajectory.frames {
        let line = frame_json(frame);
        writeln!(output, "{line}").map_err(|e| format!("write {}: {e}", traj_path.display()))?;
    }
    Ok(())
}

fn frame_json(frame: &Frame) -> String {
    let mut result = format!(
        "{{\"step\":{},\"t\":{},\"pos\":[",
        frame.step, frame.t
    );
    write_vec2s(&mut result, &frame.pos);
    result.push_str("],\"vel\":[");
    write_vec2s(&mut result, &frame.vel);
    let _ = write!(
        result,
        "],\"E_pot\":{},\"E_kin\":{}}}",
        frame.e_pot, frame.e_kin
    );
    result
}

fn write_vec2s(output: &mut String, values: &[[f64; 2]]) {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(output, "[{},{}]", value[0], value[1]);
    }
}

pub fn read(path: &Path) -> Result<Trajectory, String> {
    let dir = if path.is_dir() {
        path
    } else {
        path.parent()
            .ok_or_else(|| format!("{} has no parent directory", path.display()))?
    };
    let run_path = dir.join(RUN_FILE);
    let run_text =
        std::fs::read_to_string(&run_path).map_err(|e| format!("read {}: {e}", run_path.display()))?;
    let meta = parse_metadata(&run_text).map_err(|e| format!("parse {}: {e}", run_path.display()))?;
    let traj_path = if path.is_file() {
        path.to_path_buf()
    } else {
        dir.join(TRAJECTORY_FILE)
    };
    let input =
        File::open(&traj_path).map_err(|e| format!("open {}: {e}", traj_path.display()))?;
    let mut frames = Vec::new();
    for (index, line) in BufReader::new(input).lines().enumerate() {
        let line =
            line.map_err(|e| format!("read {} line {}: {e}", traj_path.display(), index + 1))?;
        if line.trim().is_empty() {
            return Err(format!("{} line {} is empty", traj_path.display(), index + 1));
        }
        frames.push(
            parse_frame(&line)
                .map_err(|e| format!("parse {} line {}: {e}", traj_path.display(), index + 1))?,
        );
    }
    let result = Trajectory { meta, frames };
    validate(&result)?;
    Ok(result)
}

fn parse_metadata(text: &str) -> Result<RunMetadata, String> {
    let box_values = parse_numbers(field(text, "box")?)?;
    if box_values.len() != 2 {
        return Err("box must have two numbers".into());
    }
    Ok(RunMetadata {
        n: parse(field(text, "n")?, "n")?,
        rho: parse(field(text, "rho")?, "rho")?,
        box_size: [box_values[0], box_values[1]],
        dt: parse(field(text, "dt")?, "dt")?,
        temperature: parse(field(text, "temperature")?, "temperature")?,
        eq_steps: parse(field(text, "eq_steps")?, "eq_steps")?,
        steps: parse(field(text, "steps")?, "steps")?,
        sample_every: parse(field(text, "sample_every")?, "sample_every")?,
        seed: parse(field(text, "seed")?, "seed")?,
        integrator: parse_string(field(text, "integrator")?)?,
        force: ForceStrategy::parse(&parse_string(field(text, "force")?)?)?,
        ramp_to: optional_field(text, "ramp_to")
            .map(|value| parse(value, "ramp_to"))
            .transpose()?,
    })
}

fn parse_frame(text: &str) -> Result<Frame, String> {
    if !text.trim().starts_with('{') || !text.trim().ends_with('}') {
        return Err("frame must be one JSON object".into());
    }
    Ok(Frame {
        step: parse(field(text, "step")?, "step")?,
        t: parse(field(text, "t")?, "t")?,
        pos: parse_vec2s(field(text, "pos")?)?,
        vel: parse_vec2s(field(text, "vel")?)?,
        e_pot: parse(field(text, "E_pot")?, "E_pot")?,
        e_kin: parse(field(text, "E_kin")?, "E_kin")?,
    })
}

fn parse<T: std::str::FromStr>(text: &str, name: &str) -> Result<T, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("{name} has invalid value {text:?}"))
}

fn parse_string(text: &str) -> Result<String, String> {
    let value = text.trim();
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        Ok(value[1..value.len() - 1].to_owned())
    } else {
        Err(format!("expected JSON string, got {value:?}"))
    }
}

fn parse_numbers(text: &str) -> Result<Vec<f64>, String> {
    let mut result = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        let numeric = character.is_ascii_digit() || matches!(character, '+' | '-' | '.' | 'e' | 'E');
        if numeric && start.is_none() {
            start = Some(index);
        } else if !numeric {
            if let Some(begin) = start.take() {
                result.push(parse(&text[begin..index], "array number")?);
            }
        }
    }
    if let Some(begin) = start {
        result.push(parse(&text[begin..], "array number")?);
    }
    Ok(result)
}

fn parse_vec2s(text: &str) -> Result<Vec<[f64; 2]>, String> {
    let numbers = parse_numbers(text)?;
    if numbers.len() % 2 != 0 {
        return Err("2D vector array contains an odd number of components".into());
    }
    Ok(numbers.chunks_exact(2).map(|x| [x[0], x[1]]).collect())
}

fn optional_field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    field(text, key).ok()
}

fn field<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let needle = format!("\"{key}\"");
    let key_start = text.find(&needle).ok_or_else(|| format!("missing field {key:?}"))?;
    let after_key = &text[key_start + needle.len()..];
    let colon = after_key.find(':').ok_or_else(|| format!("field {key:?} has no colon"))?;
    let value = &after_key[colon + 1..];
    let start = value
        .char_indices()
        .find(|(_, c)| !c.is_whitespace())
        .map(|(i, _)| i)
        .ok_or_else(|| format!("field {key:?} has no value"))?;
    let mut depth = 0isize;
    let mut in_string = false;
    let mut escaped = false;
    for (relative, character) in value[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '[' | '{' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => return Ok(value[start..start + relative].trim()),
            '}' if depth == 0 => return Ok(value[start..start + relative].trim()),
            _ => {}
        }
    }
    let result = value[start..].trim();
    if result.is_empty() {
        Err(format!("field {key:?} has no value"))
    } else {
        Ok(result)
    }
}

pub fn validate(trajectory: &Trajectory) -> Result<(), String> {
    let m = &trajectory.meta;
    if m.n == 0
        || !m.rho.is_finite()
        || m.rho <= 0.0
        || !m.dt.is_finite()
        || m.dt <= 0.0
        || !m.temperature.is_finite()
        || m.temperature <= 0.0
        || m.sample_every == 0
        || m.box_size.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("run.json contains invalid physical metadata".into());
    }
    if m.integrator != "velocity-verlet" && m.integrator != "euler" {
        return Err(format!("unknown integrator {:?}", m.integrator));
    }
    let expected = m.steps / m.sample_every;
    if trajectory.frames.len() != expected {
        return Err(format!(
            "expected {expected} frames from steps/sample_every, found {}",
            trajectory.frames.len()
        ));
    }
    for (index, frame) in trajectory.frames.iter().enumerate() {
        let expected_step = (index + 1) * m.sample_every;
        if frame.step != expected_step {
            return Err(format!(
                "frame {} has step {}, expected {expected_step}",
                index + 1,
                frame.step
            ));
        }
        let expected_t = frame.step as f64 * m.dt;
        if !frame.t.is_finite()
            || (frame.t - expected_t).abs() > 1e-10 * expected_t.max(1.0)
        {
            return Err(format!(
                "frame {} has t={}, expected {expected_t}",
                index + 1,
                frame.t
            ));
        }
        if frame.pos.len() != m.n || frame.vel.len() != m.n {
            return Err(format!("frame {} does not contain {} atoms", index + 1, m.n));
        }
        if !frame.e_pot.is_finite() || !frame.e_kin.is_finite() {
            return Err(format!("frame {} has non-finite energy", index + 1));
        }
        for (atom, (p, v)) in frame.pos.iter().zip(&frame.vel).enumerate() {
            if p.iter().chain(v).any(|x| !x.is_finite())
                || !(0.0..m.box_size[0]).contains(&p[0])
                || !(0.0..m.box_size[1]).contains(&p[1])
            {
                return Err(format!("frame {} atom {atom} has invalid state", index + 1));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Trajectory {
        Trajectory {
            meta: RunMetadata {
                n: 2,
                rho: 0.02,
                box_size: [10.0, 10.0],
                dt: 0.01,
                temperature: 0.5,
                eq_steps: 2,
                steps: 2,
                sample_every: 1,
                seed: 2026,
                integrator: "velocity-verlet".into(),
                force: ForceStrategy::Cells,
                ramp_to: None,
            },
            frames: vec![
                Frame {
                    step: 1,
                    t: 0.01,
                    pos: vec![[1.0, 1.0], [3.0, 3.0]],
                    vel: vec![[0.1, 0.2], [-0.1, -0.2]],
                    e_pot: -0.1,
                    e_kin: 0.05,
                },
                Frame {
                    step: 2,
                    t: 0.02,
                    pos: vec![[1.01, 1.02], [2.99, 2.98]],
                    vel: vec![[0.1, 0.2], [-0.1, -0.2]],
                    e_pot: -0.1,
                    e_kin: 0.05,
                },
            ],
        }
    }

    #[test]
    fn json_jsonl_round_trip() {
        let dir = std::env::temp_dir().join(format!("md-json-roundtrip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write(&dir, &sample()).unwrap();
        let read_back = read(&dir).unwrap();
        assert_eq!(read_back.meta.box_size, [10.0, 10.0]);
        assert_eq!(read_back.frames.len(), 2);
        assert!(std::fs::read_to_string(dir.join(TRAJECTORY_FILE))
            .unwrap()
            .lines()
            .all(|line| line.starts_with('{') && line.ends_with('}')));
    }

    #[test]
    fn malformed_frame_count_or_json_is_rejected() {
        let mut bad = sample();
        bad.frames.pop();
        assert!(validate(&bad).is_err());
        let dir = std::env::temp_dir().join(format!("md-json-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write(&dir, &sample()).unwrap();
        std::fs::write(dir.join(TRAJECTORY_FILE), "{bad json}\n").unwrap();
        assert!(read(&dir).is_err());
    }
}
