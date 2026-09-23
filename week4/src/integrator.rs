pub trait Integrator: Send + Sync {
    fn step(&self, state: &mut [f64], dt: f64, rate: &dyn Fn(&[f64], &mut [f64]));
}

pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn step(&self, state: &mut [f64], dt: f64, rate: &dyn Fn(&[f64], &mut [f64])) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        rate(state, &mut k1);
        for i in 0..n {
            state[i] += dt * k1[i];
        }
    }
}

pub struct ExplicitMidpoint;

impl Integrator for ExplicitMidpoint {
    fn step(&self, state: &mut [f64], dt: f64, rate: &dyn Fn(&[f64], &mut [f64])) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        rate(state, &mut k1);

        let mut u_mid = vec![0.0; n];
        for i in 0..n {
            u_mid[i] = state[i] + 0.5 * dt * k1[i];
        }

        let mut k2 = vec![0.0; n];
        rate(&u_mid, &mut k2);

        for i in 0..n {
            state[i] += dt * k2[i];
        }
    }
}

pub struct Rk4;

impl Integrator for Rk4 {
    fn step(&self, state: &mut [f64], dt: f64, rate: &dyn Fn(&[f64], &mut [f64])) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        rate(state, &mut k1);

        let mut u_temp = vec![0.0; n];
        for i in 0..n {
            u_temp[i] = state[i] + 0.5 * dt * k1[i];
        }
        let mut k2 = vec![0.0; n];
        rate(&u_temp, &mut k2);

        for i in 0..n {
            u_temp[i] = state[i] + 0.5 * dt * k2[i];
        }
        let mut k3 = vec![0.0; n];
        rate(&u_temp, &mut k3);

        for i in 0..n {
            u_temp[i] = state[i] + dt * k3[i];
        }
        let mut k4 = vec![0.0; n];
        rate(&u_temp, &mut k4);

        for i in 0..n {
            state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
}

pub struct Rk4EqualWeights;

impl Integrator for Rk4EqualWeights {
    fn step(&self, state: &mut [f64], dt: f64, rate: &dyn Fn(&[f64], &mut [f64])) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        rate(state, &mut k1);

        let mut u_temp = vec![0.0; n];
        for i in 0..n {
            u_temp[i] = state[i] + 0.5 * dt * k1[i];
        }
        let mut k2 = vec![0.0; n];
        rate(&u_temp, &mut k2);

        for i in 0..n {
            u_temp[i] = state[i] + 0.5 * dt * k2[i];
        }
        let mut k3 = vec![0.0; n];
        rate(&u_temp, &mut k3);

        for i in 0..n {
            u_temp[i] = state[i] + dt * k3[i];
        }
        let mut k4 = vec![0.0; n];
        rate(&u_temp, &mut k4);

        for i in 0..n {
            state[i] += (dt / 4.0) * (k1[i] + k2[i] + k3[i] + k4[i]);
        }
    }
}
