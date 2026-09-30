use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawExperiment {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[f64; 2]>,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
    pub background: Vec<Vec<f64>>,
    pub perturbation: Option<Vec<Vec<f64>>>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[usize; 2]>,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
    pub background: Vec<f64>,        // flattened [z, x]
    pub perturbation: Option<Vec<f64>>, // flattened [z, x]
    pub sigma: Vec<f64>,             // flattened [z, x]
    pub raw_meta: serde_json::Value, // json object without background and perturbation
}

impl Experiment {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut raw_val: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let raw: RawExperiment = serde_json::from_value(raw_val.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        if let serde_json::Value::Object(ref mut map) = raw_val {
            map.remove("background");
            map.remove("perturbation");
        }

        let nx = raw.nx;
        let nz = raw.nz;
        let mut bg_flat = vec![0.0; nz * nx];
        for z in 0..nz {
            for x in 0..nx {
                bg_flat[z * nx + x] = raw.background[z][x];
            }
        }

        let pert_flat = raw.perturbation.as_ref().map(|p| {
            let mut flat = vec![0.0; nz * nx];
            for z in 0..nz {
                for x in 0..nx {
                    flat[z * nx + x] = p[z][x];
                }
            }
            flat
        });

        // Compute sponge sigma
        let mut sigma = vec![0.0; nz * nx];
        let w = raw.sponge_width as f64;
        let s_max = raw.sponge_strength;
        if raw.sponge_width > 0 {
            for z in 0..nz {
                for x in 0..nx {
                    let lz = z.min(nz - 1 - z);
                    let lx = x.min(nx - 1 - x);
                    let l = (lz.min(lx)) as f64;
                    if l < w {
                        let ratio = 1.0 - l / w;
                        sigma[z * nx + x] = s_max * ratio * ratio;
                    }
                }
            }
        }

        let receivers: Vec<[usize; 2]> = raw.receivers.iter().map(|&[rx, rz]| [rx.round() as usize, rz.round() as usize]).collect();

        Ok(Self {
            nx,
            nz,
            dx: raw.dx,
            dt: raw.dt,
            steps: raw.steps,
            source_frequency: raw.source_frequency,
            source_peak_time: raw.source_peak_time,
            source_amplitude: raw.source_amplitude,
            shots: raw.shots,
            receivers,
            sponge_width: raw.sponge_width,
            sponge_strength: raw.sponge_strength,
            length_unit_m: raw.length_unit_m,
            time_unit_s: raw.time_unit_s,
            background: bg_flat,
            perturbation: pert_flat,
            sigma,
            raw_meta: raw_val,
        })
    }

    pub fn compute_source_footprint(&self, shot_idx: usize) -> Vec<f64> {
        let [xs, zs] = self.shots[shot_idx];
        let mut fp = vec![0.0; self.nz * self.nx];
        for z in 0..self.nz {
            let dz = z as f64 - zs;
            for x in 0..self.nx {
                let dx = x as f64 - xs;
                let r2 = dx * dx + dz * dz;
                fp[z * self.nx + x] = (-r2 / 2.0).exp();
            }
        }
        fp
    }

    pub fn ricker_pulse(&self, step: usize) -> f64 {
        let t = step as f64 * self.dt;
        let theta = std::f64::consts::PI * self.source_frequency * (t - self.source_peak_time);
        self.source_amplitude * (1.0 - 2.0 * theta * theta) * (-theta * theta).exp()
    }
}
