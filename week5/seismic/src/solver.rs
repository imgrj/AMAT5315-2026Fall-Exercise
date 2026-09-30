use crate::experiment::Experiment;
use crate::treeverse::TreeverseAction;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CompleteState {
    pub u_prev: Vec<f64>,
    pub u_curr: Vec<f64>,
    pub step: usize,
}

impl CompleteState {
    pub fn new_zero(nz: usize, nx: usize, step: usize) -> Self {
        Self {
            u_prev: vec![0.0; nz * nx],
            u_curr: vec![0.0; nz * nx],
            step,
        }
    }

    pub fn size_bytes(&self) -> usize {
        (self.u_prev.len() + self.u_curr.len()) * std::mem::size_of::<f64>()
    }
}

pub fn laplacian(nx: usize, nz: usize, dx: f64, u: &[f64], lap: &mut [f64]) {
    let inv_dx2 = 1.0 / (dx * dx);
    for z in 1..nz - 1 {
        let row = z * nx;
        let row_up = (z - 1) * nx;
        let row_down = (z + 1) * nx;
        for x in 1..nx - 1 {
            let u_c = u[row + x];
            let u_l = u[row + x - 1];
            let u_r = u[row + x + 1];
            let u_u = u[row_up + x];
            let u_d = u[row_down + x];
            lap[row + x] = (u_r + u_l + u_u + u_d - 4.0 * u_c) * inv_dx2;
        }
    }
}

pub fn forward_step(
    exp: &Experiment,
    c: &[f64],
    footprint: Option<&[f64]>,
    state: &CompleteState,
    lap_buf: &mut [f64],
) -> CompleteState {
    let nx = exp.nx;
    let nz = exp.nz;
    let dt = exp.dt;
    let dt2 = dt * dt;
    let n = state.step;

    laplacian(nx, nz, exp.dx, &state.u_curr, lap_buf);

    let pulse = if footprint.is_some() {
        exp.ricker_pulse(n)
    } else {
        0.0
    };

    let mut u_next = vec![0.0; nz * nx];
    for z in 1..nz - 1 {
        let row = z * nx;
        for x in 1..nx - 1 {
            let idx = row + x;
            let sigma = exp.sigma[idx];
            let den = 1.0 + sigma * dt;
            let q = if let Some(fp) = footprint {
                pulse * fp[idx]
            } else {
                0.0
            };
            let c_val = c[idx];
            let num = 2.0 * state.u_curr[idx] - (1.0 - sigma * dt) * state.u_prev[idx]
                + dt2 * (c_val * c_val * lap_buf[idx] + q);
            u_next[idx] = num / den;
        }
    }

    CompleteState {
        u_prev: state.u_curr.clone(),
        u_curr: u_next,
        step: n + 1,
    }
}

pub fn run_forward_shot(
    exp: &Experiment,
    c: &[f64],
    shot_idx: usize,
    every: Option<usize>,
) -> (Vec<f64>, Option<Vec<f32>>, Option<Vec<usize>>, Option<Vec<f64>>) {
    let nx = exp.nx;
    let nz = exp.nz;
    let steps = exp.steps;
    let n_recvs = exp.receivers.len();

    let footprint = exp.compute_source_footprint(shot_idx);
    let mut state = CompleteState::new_zero(nz, nx, 0);
    let mut lap_buf = vec![0.0; nz * nx];

    let mut traces = vec![0.0; steps * n_recvs];

    let record_frames = every.is_some();
    let mut recorded_frames: Vec<f32> = Vec::new();
    let mut frame_steps: Vec<usize> = Vec::new();
    let mut frame_times: Vec<f64> = Vec::new();

    if let Some(ev) = every {
        frame_steps.push(0);
        frame_times.push(0.0);
        for &val in &state.u_curr {
            recorded_frames.push(val as f32);
        }

        for n in 0..steps {
            state = forward_step(exp, c, Some(&footprint), &state, &mut lap_buf);

            for (k, &[rx, rz]) in exp.receivers.iter().enumerate() {
                traces[n * n_recvs + k] = state.u_curr[rz * nx + rx];
            }

            if (n + 1) % ev == 0 {
                frame_steps.push(n + 1);
                frame_times.push((n + 1) as f64 * exp.dt);
                for &val in &state.u_curr {
                    recorded_frames.push(val as f32);
                }
            }
        }
    } else {
        for n in 0..steps {
            state = forward_step(exp, c, Some(&footprint), &state, &mut lap_buf);
            for (k, &[rx, rz]) in exp.receivers.iter().enumerate() {
                traces[n * n_recvs + k] = state.u_curr[rz * nx + rx];
            }
        }
    }

    if record_frames {
        (traces, Some(recorded_frames), Some(frame_steps), Some(frame_times))
    } else {
        (traces, None, None, None)
    }
}

pub fn run_born_shot(
    exp: &Experiment,
    shot_idx: usize,
) -> Vec<f64> {
    let nx = exp.nx;
    let nz = exp.nz;
    let dt = exp.dt;
    let dt2 = dt * dt;
    let steps = exp.steps;
    let n_recvs = exp.receivers.len();

    let footprint = exp.compute_source_footprint(shot_idx);
    let c = &exp.background;
    let m = exp.perturbation.as_ref().expect("Born mode requires perturbation");

    let mut u_prev = vec![0.0; nz * nx];
    let mut u_curr = vec![0.0; nz * nx];
    let mut du_prev = vec![0.0; nz * nx];
    let mut du_curr = vec![0.0; nz * nx];

    let mut lap_u = vec![0.0; nz * nx];
    let mut lap_du = vec![0.0; nz * nx];

    let mut born_traces = vec![0.0; steps * n_recvs];

    for n in 0..steps {
        let pulse = exp.ricker_pulse(n);
        laplacian(nx, nz, exp.dx, &u_curr, &mut lap_u);
        laplacian(nx, nz, exp.dx, &du_curr, &mut lap_du);

        let mut u_next = vec![0.0; nz * nx];
        let mut du_next = vec![0.0; nz * nx];

        for z in 1..nz - 1 {
            let row = z * nx;
            for x in 1..nx - 1 {
                let idx = row + x;
                let sigma = exp.sigma[idx];
                let den = 1.0 + sigma * dt;
                let q = pulse * footprint[idx];
                let c_val = c[idx];

                let num_u = 2.0 * u_curr[idx] - (1.0 - sigma * dt) * u_prev[idx]
                    + dt2 * (c_val * c_val * lap_u[idx] + q);
                u_next[idx] = num_u / den;

                let num_du = 2.0 * du_curr[idx] - (1.0 - sigma * dt) * du_prev[idx]
                    + dt2 * (c_val * c_val * lap_du[idx] + 2.0 * c_val * m[idx] * lap_u[idx]);
                du_next[idx] = num_du / den;
            }
        }

        for (k, &[rx, rz]) in exp.receivers.iter().enumerate() {
            born_traces[n * n_recvs + k] = du_next[rz * nx + rx];
        }

        u_prev = u_curr;
        u_curr = u_next;

        du_prev = du_curr;
        du_curr = du_next;
    }

    born_traces
}

pub struct AdjointShotResult {
    pub image_contrib: Vec<f64>,
    pub reverse_calls: usize,
    pub forward_calls: usize,
    pub peak_saved_states: usize,
    pub recorded_frames: Option<Vec<f32>>,
    pub frame_steps: Option<Vec<usize>>,
    pub frame_times: Option<Vec<f64>>,
}

/// Run Adjoint for a single shot with full history
pub fn run_adjoint_shot_full(
    exp: &Experiment,
    shot_idx: usize,
    weights: &[f64], // [steps, n_recvs]
    every: Option<usize>,
) -> AdjointShotResult {
    let nx = exp.nx;
    let nz = exp.nz;
    let dt = exp.dt;
    let dt2 = dt * dt;
    let steps = exp.steps;
    let n_recvs = exp.receivers.len();

    let footprint = exp.compute_source_footprint(shot_idx);
    let c = &exp.background;

    // 1. Forward run keeping full trajectory
    let mut history: Vec<CompleteState> = Vec::with_capacity(steps + 1);
    let mut state = CompleteState::new_zero(nz, nx, 0);
    history.push(state.clone());

    let mut lap_buf = vec![0.0; nz * nx];
    for _ in 0..steps {
        state = forward_step(exp, c, Some(&footprint), &state, &mut lap_buf);
        history.push(state.clone());
    }

    // 2. Reverse pass
    let mut image_contrib = vec![0.0; nz * nx];
    let mut adj_curr = vec![0.0; nz * nx];
    let mut adj_next = vec![0.0; nz * nx];

    let mut v = vec![0.0; nz * nx];
    let mut c2_v = vec![0.0; nz * nx];
    let mut lap_c2_v = vec![0.0; nz * nx];
    let mut lap_un = vec![0.0; nz * nx];

    let mut recorded_frames: Vec<f32> = Vec::new();
    let mut frame_steps: Vec<usize> = Vec::new();
    let mut frame_times: Vec<f64> = Vec::new();

    for n in (0..steps).rev() {
        // Receivers re-emit into adj_next
        for (k, &[rx, rz]) in exp.receivers.iter().enumerate() {
            adj_next[rz * nx + rx] += weights[n * n_recvs + k];
        }

        // v = adj_next / (1 + sigma*dt)
        for z in 1..nz - 1 {
            let row = z * nx;
            for x in 1..nx - 1 {
                let idx = row + x;
                v[idx] = adj_next[idx] / (1.0 + exp.sigma[idx] * dt);
                c2_v[idx] = c[idx] * c[idx] * v[idx];
            }
        }

        // Image accumulation from u^n (which is in history[n].u_curr)
        let u_n = &history[n].u_curr;
        laplacian(nx, nz, exp.dx, u_n, &mut lap_un);
        for z in 1..nz - 1 {
            let row = z * nx;
            for x in 1..nx - 1 {
                let idx = row + x;
                image_contrib[idx] += 2.0 * c[idx] * dt2 * lap_un[idx] * v[idx];
            }
        }

        // Adjoint propagation
        laplacian(nx, nz, exp.dx, &c2_v, &mut lap_c2_v);
        let mut adj_prev = vec![0.0; nz * nx];
        for z in 1..nz - 1 {
            let row = z * nx;
            for x in 1..nx - 1 {
                let idx = row + x;
                adj_curr[idx] += 2.0 * v[idx] + dt2 * lap_c2_v[idx];
                adj_prev[idx] = -(1.0 - exp.sigma[idx] * dt) * v[idx];
            }
        }

        // Recording if requested: adjoint of u^n after reversing step n
        if let Some(ev) = every {
            if n % ev == 0 {
                frame_steps.push(n);
                frame_times.push(n as f64 * dt);
                for &val in &adj_curr {
                    recorded_frames.push(val as f32);
                }
            }
        }

        // Shift for step n-1
        adj_next = adj_curr;
        adj_curr = adj_prev;
    }

    AdjointShotResult {
        image_contrib,
        reverse_calls: steps,
        forward_calls: steps,
        peak_saved_states: steps + 1,
        recorded_frames: if every.is_some() { Some(recorded_frames) } else { None },
        frame_steps: if every.is_some() { Some(frame_steps) } else { None },
        frame_times: if every.is_some() { Some(frame_times) } else { None },
    }
}

/// Run Adjoint for a single shot with Treeverse checkpointing
pub fn run_adjoint_shot_treeverse(
    exp: &Experiment,
    shot_idx: usize,
    weights: &[f64],
    _delta: usize,
    actions: &[TreeverseAction],
    every: Option<usize>,
) -> AdjointShotResult {
    let nx = exp.nx;
    let nz = exp.nz;
    let dt = exp.dt;
    let dt2 = dt * dt;
    let n_recvs = exp.receivers.len();

    let footprint = exp.compute_source_footprint(shot_idx);
    let c = &exp.background;

    let mut saved_slots: HashMap<usize, CompleteState> = HashMap::new();
    saved_slots.insert(0, CompleteState::new_zero(nz, nx, 0));

    let mut working_state = CompleteState::new_zero(nz, nx, 0);
    let mut lap_buf = vec![0.0; nz * nx];

    let mut image_contrib = vec![0.0; nz * nx];
    let mut adj_curr = vec![0.0; nz * nx];
    let mut adj_next = vec![0.0; nz * nx];

    let mut v = vec![0.0; nz * nx];
    let mut c2_v = vec![0.0; nz * nx];
    let mut lap_c2_v = vec![0.0; nz * nx];
    let mut lap_un = vec![0.0; nz * nx];

    let mut recorded_frames: Vec<f32> = Vec::new();
    let mut frame_steps: Vec<usize> = Vec::new();
    let mut frame_times: Vec<f64> = Vec::new();

    let mut forward_calls = 0;
    let mut reverse_calls = 0;
    let mut peak_saved = 1;

    for act in actions {
        match act.action.as_str() {
            "store" => {
                saved_slots.insert(act.step, working_state.clone());
                peak_saved = peak_saved.max(saved_slots.len());
            }
            "restore" => {
                working_state = saved_slots.get(&act.step).expect("restore slot").clone();
            }
            "call" => {
                forward_calls += 1;
                working_state = forward_step(exp, c, Some(&footprint), &working_state, &mut lap_buf);
            }
            "grad" => {
                reverse_calls += 1;
                let n = act.step;
                let s_n = saved_slots.get(&n).expect("grad slot");

                for (k, &[rx, rz]) in exp.receivers.iter().enumerate() {
                    adj_next[rz * nx + rx] += weights[n * n_recvs + k];
                }

                for z in 1..nz - 1 {
                    let row = z * nx;
                    for x in 1..nx - 1 {
                        let idx = row + x;
                        v[idx] = adj_next[idx] / (1.0 + exp.sigma[idx] * dt);
                        c2_v[idx] = c[idx] * c[idx] * v[idx];
                    }
                }

                let u_n = &s_n.u_curr;
                laplacian(nx, nz, exp.dx, u_n, &mut lap_un);
                for z in 1..nz - 1 {
                    let row = z * nx;
                    for x in 1..nx - 1 {
                        let idx = row + x;
                        image_contrib[idx] += 2.0 * c[idx] * dt2 * lap_un[idx] * v[idx];
                    }
                }

                laplacian(nx, nz, exp.dx, &c2_v, &mut lap_c2_v);
                let mut adj_prev = vec![0.0; nz * nx];
                for z in 1..nz - 1 {
                    let row = z * nx;
                    for x in 1..nx - 1 {
                        let idx = row + x;
                        adj_curr[idx] += 2.0 * v[idx] + dt2 * lap_c2_v[idx];
                        adj_prev[idx] = -(1.0 - exp.sigma[idx] * dt) * v[idx];
                    }
                }

                if let Some(ev) = every {
                    if n % ev == 0 {
                        frame_steps.push(n);
                        frame_times.push(n as f64 * dt);
                        for &val in &adj_curr {
                            recorded_frames.push(val as f32);
                        }
                    }
                }

                adj_next = adj_curr;
                adj_curr = adj_prev;
            }
            "fetch" => {
                saved_slots.remove(&act.step);
            }
            _ => {}
        }
    }

    AdjointShotResult {
        image_contrib,
        reverse_calls,
        forward_calls,
        peak_saved_states: peak_saved,
        recorded_frames: if every.is_some() { Some(recorded_frames) } else { None },
        frame_steps: if every.is_some() { Some(frame_steps) } else { None },
        frame_times: if every.is_some() { Some(frame_times) } else { None },
    }
}
