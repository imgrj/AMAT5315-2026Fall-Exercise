//! The learned branching rule: one shared MLP over the 12 per-spin features of
//! every candidate, compared with a masked softmax (Eq. 8-9 of the sheet).
//!
//! `artifacts/model.json` is written by `train_branch.py`; this module reads it,
//! reproduces the JAX probabilities on the six saved probes, and scores spins
//! inside Branch-and-Bound.

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct Probe {
    pub instance: usize,
    pub node: usize,
    pub features: Vec<Vec<f64>>,
    pub logits: Vec<f64>,
    pub probs: Vec<f64>,
}

#[derive(Deserialize, Debug)]
struct NetJson {
    mean: Vec<f64>,
    std: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    hidden: usize,
    probes: Vec<Probe>,
    #[serde(default)]
    best_epoch: Option<i64>,
    #[serde(default)]
    best_val_ce: Option<f64>,
}

pub struct Net {
    mean: Vec<f64>,
    std: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    hidden: usize,
    pub probes: Vec<Probe>,
    pub best_epoch: Option<i64>,
    pub best_val_ce: Option<f64>,
}

impl Net {
    pub fn from_json(text: &str) -> Result<Net, String> {
        let j: NetJson = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let n_in = j.mean.len();
        if n_in != crate::bnb::N_FEATURES || j.std.len() != n_in {
            return Err("feature count mismatch".to_string());
        }
        if j.w1.len() != j.hidden * n_in || j.b1.len() != j.hidden || j.w2.len() != j.hidden {
            return Err("weight shape mismatch".to_string());
        }
        Ok(Net {
            mean: j.mean,
            std: j.std,
            w1: j.w1,
            b1: j.b1,
            w2: j.w2,
            b2: j.b2,
            hidden: j.hidden,
            probes: j.probes,
            best_epoch: j.best_epoch,
            best_val_ce: j.best_val_ce,
        })
    }

    /// The standardized input of feature vector `phi`.
    fn standardize(&self, phi: &[f64]) -> Vec<f64> {
        phi.iter()
            .enumerate()
            .map(|(k, v)| (v - self.mean[k]) / self.std[k].max(1e-6))
            .collect()
    }

    /// `f_theta(phi) = w2 . tanh(W1 phi + b1) + b2`.
    pub fn score(&self, phi: &[f64; crate::bnb::N_FEATURES]) -> f64 {
        let z = self.standardize(phi);
        let mut out = self.b2;
        for h in 0..self.hidden {
            let row = &self.w1[h * z.len()..(h + 1) * z.len()];
            let mut acc = self.b1[h];
            for k in 0..z.len() {
                acc += row[k] * z[k];
            }
            out += self.w2[h] * acc.tanh();
        }
        out
    }

    /// Scores of every spin at the root, standardized to mean 0 and std 1.
    pub fn root_scores(&self, all: &[[f64; crate::bnb::N_FEATURES]]) -> Vec<f64> {
        let raw: Vec<f64> = all.iter().map(|p| self.score(p)).collect();
        let n = raw.len() as f64;
        let mean = raw.iter().sum::<f64>() / n;
        let var = raw.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / n;
        let sd = var.sqrt().max(1e-6);
        raw.iter().map(|r| (r - mean) / sd).collect()
    }

    /// Logits of every candidate at one node.
    pub fn logits(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|phi| {
                let mut arr = [0.0f64; crate::bnb::N_FEATURES];
                arr.copy_from_slice(&phi[..crate::bnb::N_FEATURES]);
                self.score(&arr)
            })
            .collect()
    }

    /// Masked softmax over the candidate spins present at a node.
    pub fn probabilities(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let logits = self.logits(candidates);
        softmax(&logits)
    }

    /// Largest absolute probability difference between this implementation and
    /// the six JAX probes, and the largest logit difference.
    pub fn probe_differences(&self) -> (f64, f64) {
        let mut dp = 0.0f64;
        let mut dl = 0.0f64;
        for p in &self.probes {
            let logits = self.logits(&p.features);
            let probs = self.probabilities(&p.features);
            for (a, b) in logits.iter().zip(p.logits.iter()) {
                dl = dl.max((a - b).abs());
            }
            for (a, b) in probs.iter().zip(p.probs.iter()) {
                dp = dp.max((a - b).abs());
            }
        }
        (dp, dl)
    }
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}
