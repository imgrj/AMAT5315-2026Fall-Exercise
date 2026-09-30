use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeverseAction {
    pub action: String,
    pub step: usize,
    pub saved_states: usize,
}

pub fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut c = 1u128;
    for i in 0..k {
        c = c * (n - i) as u128 / (i + 1) as u128;
    }
    c
}

pub fn binomial_fit(n_steps: usize, delta: usize) -> usize {
    let mut tau = 1;
    while (n_steps as u128) > binomial(tau + delta, delta) {
        tau += 1;
    }
    tau
}

pub fn mid(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    let den = tau + delta;
    if den == 0 {
        return sigma;
    }
    let num = delta * sigma + tau * phi;
    let mut kappa = (num + den - 1) / den;
    if kappa >= phi && delta > 0 {
        kappa = (sigma + 1).max(phi.saturating_sub(1));
    }
    kappa
}

pub struct ScheduleRecorder {
    pub actions: Vec<TreeverseAction>,
    pub saved_count: usize,
    pub peak_saved: usize,
    pub forward_calls: usize,
    pub reverse_calls: usize,
    pub saved_slots: HashSet<usize>,
    pub delta: usize,
}

impl ScheduleRecorder {
    pub fn new(delta: usize) -> Self {
        let mut slots = HashSet::new();
        slots.insert(0);
        Self {
            actions: Vec::new(),
            saved_count: 1,
            peak_saved: 1,
            forward_calls: 0,
            reverse_calls: 0,
            saved_slots: slots,
            delta,
        }
    }

    pub fn record(&mut self, action: &str, step: usize) {
        match action {
            "store" => {
                self.saved_count += 1;
                self.saved_slots.insert(step);
            }
            "fetch" => {
                self.saved_count -= 1;
                self.saved_slots.remove(&step);
            }
            "call" => {
                self.forward_calls += 1;
            }
            "grad" => {
                self.reverse_calls += 1;
            }
            _ => {}
        }
        if self.saved_count > self.peak_saved {
            self.peak_saved = self.saved_count;
        }
        self.actions.push(TreeverseAction {
            action: action.to_string(),
            step,
            saved_states: self.saved_count,
        });
    }
}

pub fn build_treeverse_schedule(n_steps: usize, delta: usize) -> ScheduleRecorder {
    let tau = binomial_fit(n_steps, delta);
    let mut recorder = ScheduleRecorder::new(delta);
    let mut working_pos = 0;

    fn recurse(
        delta_cur: usize,
        mut tau_cur: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
        working_pos: &mut usize,
        recorder: &mut ScheduleRecorder,
    ) {
        let effective_delta = if sigma > beta {
            let next_delta = delta_cur.saturating_sub(1);
            if *working_pos != beta {
                recorder.record("restore", beta);
                *working_pos = beta;
            }
            for j in beta..sigma {
                recorder.record("call", j);
                *working_pos = j + 1;
            }
            recorder.record("store", sigma);
            next_delta
        } else {
            delta_cur
        };

        let mut kappa = mid(effective_delta, tau_cur, sigma, phi);
        while tau_cur > 0 && kappa < phi {
            recurse(effective_delta, tau_cur, sigma, kappa, phi, working_pos, recorder);
            tau_cur -= 1;
            phi = kappa;
            if tau_cur > 0 {
                kappa = mid(effective_delta, tau_cur, sigma, phi);
            }
        }

        recorder.record("grad", sigma);
        if sigma > beta {
            recorder.record("fetch", sigma);
        }
    }

    recurse(delta, tau, 0, 0, n_steps, &mut working_pos, &mut recorder);
    recorder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counts() {
        let s1 = build_treeverse_schedule(240, 1);
        let s3 = build_treeverse_schedule(240, 3);
        let s5 = build_treeverse_schedule(240, 5);
        let s10 = build_treeverse_schedule(240, 10);
        assert_eq!(s1.forward_calls, 28680);
        assert_eq!(s3.forward_calls, 1695);
        assert_eq!(s5.forward_calls, 990);
        assert_eq!(s10.forward_calls, 642);
    }
}
