use std::f64::consts::PI;
use num_complex::Complex64;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;
use crate::rng::Rng;

pub struct Spectral2D {
    pub n: usize,
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
}

impl Spectral2D {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(n);
        let fft_inverse = planner.plan_fft_inverse(n);
        Self {
            n,
            fft_forward,
            fft_inverse,
        }
    }

    pub fn fft2(&self, data: &[Complex64], out: &mut [Complex64]) {
        let n = self.n;
        out.copy_from_slice(data);

        for l in 0..n {
            let row = &mut out[l * n..(l + 1) * n];
            self.fft_forward.process(row);
        }

        let mut col_buf = vec![Complex64::default(); n];
        for j in 0..n {
            for l in 0..n {
                col_buf[l] = out[l * n + j];
            }
            self.fft_forward.process(&mut col_buf);
            for l in 0..n {
                out[l * n + j] = col_buf[l];
            }
        }
    }

    pub fn ifft2(&self, data: &[Complex64], out: &mut [Complex64]) {
        let n = self.n;
        out.copy_from_slice(data);

        for l in 0..n {
            let row = &mut out[l * n..(l + 1) * n];
            self.fft_inverse.process(row);
        }

        let mut col_buf = vec![Complex64::default(); n];
        for j in 0..n {
            for l in 0..n {
                col_buf[l] = out[l * n + j];
            }
            self.fft_inverse.process(&mut col_buf);
            for l in 0..n {
                out[l * n + j] = col_buf[l];
            }
        }

        let inv_n2 = 1.0 / ((n * n) as f64);
        for item in out.iter_mut() {
            *item *= inv_n2;
        }
    }

    pub fn real_to_complex(&self, real: &[f64]) -> Vec<Complex64> {
        real.iter().map(|&x| Complex64::new(x, 0.0)).collect()
    }

    pub fn complex_to_real(&self, comp: &[Complex64]) -> Vec<f64> {
        comp.iter().map(|c| c.re).collect()
    }

    #[inline]
    pub fn wavenumbers(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let mut kx = vec![0.0; n];
        let mut ky = vec![0.0; n];
        let half = n / 2;
        for j in 0..n {
            kx[j] = if j < half {
                j as f64
            } else if j == half {
                -(half as f64)
            } else {
                (j as f64) - (n as f64)
            };
        }
        for l in 0..n {
            ky[l] = if l < half {
                l as f64
            } else if l == half {
                -(half as f64)
            } else {
                (l as f64) - (n as f64)
            };
        }
        (kx, ky)
    }

    pub fn dealias(&self, hat: &mut [Complex64]) {
        let n = self.n;
        let k_max = (n / 3) as f64;
        let (kx, ky) = self.wavenumbers();
        for l in 0..n {
            let y_k = ky[l].abs();
            for j in 0..n {
                let x_k = kx[j].abs();
                if x_k > k_max || y_k > k_max {
                    hat[l * n + j] = Complex64::default();
                }
            }
        }
    }

    pub fn diff_x(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n;
        let (kx, _) = self.wavenumbers();
        let fc = self.real_to_complex(f);
        let mut fhat = vec![Complex64::default(); n * n];
        self.fft2(&fc, &mut fhat);
        for l in 0..n {
            for j in 0..n {
                let idx = l * n + j;
                fhat[idx] *= Complex64::new(0.0, kx[j]);
            }
        }
        let mut outc = vec![Complex64::default(); n * n];
        self.ifft2(&fhat, &mut outc);
        self.complex_to_real(&outc)
    }

    pub fn diff_xx(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n;
        let (kx, _) = self.wavenumbers();
        let fc = self.real_to_complex(f);
        let mut fhat = vec![Complex64::default(); n * n];
        self.fft2(&fc, &mut fhat);
        for l in 0..n {
            for j in 0..n {
                let idx = l * n + j;
                fhat[idx] *= -kx[j] * kx[j];
            }
        }
        let mut outc = vec![Complex64::default(); n * n];
        self.ifft2(&fhat, &mut outc);
        self.complex_to_real(&outc)
    }

    pub fn diff_xy(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n;
        let (kx, ky) = self.wavenumbers();
        let fc = self.real_to_complex(f);
        let mut fhat = vec![Complex64::default(); n * n];
        self.fft2(&fc, &mut fhat);
        for l in 0..n {
            for j in 0..n {
                let idx = l * n + j;
                fhat[idx] *= -kx[j] * ky[l];
            }
        }
        let mut outc = vec![Complex64::default(); n * n];
        self.ifft2(&fhat, &mut outc);
        self.complex_to_real(&outc)
    }

    pub fn laplacian(&self, f: &[f64]) -> Vec<f64> {
        let n = self.n;
        let (kx, ky) = self.wavenumbers();
        let fc = self.real_to_complex(f);
        let mut fhat = vec![Complex64::default(); n * n];
        self.fft2(&fc, &mut fhat);
        for l in 0..n {
            for j in 0..n {
                let idx = l * n + j;
                fhat[idx] *= -(kx[j] * kx[j] + ky[l] * ky[l]);
            }
        }
        let mut outc = vec![Complex64::default(); n * n];
        self.ifft2(&fhat, &mut outc);
        self.complex_to_real(&outc)
    }

    pub fn rate(&self, omega_real: &[f64], out_rate: &mut [f64], nu: f64) {
        let n = self.n;
        let (kx, ky) = self.wavenumbers();

        let omega_c = self.real_to_complex(omega_real);
        let mut omega_hat = vec![Complex64::default(); n * n];
        self.fft2(&omega_c, &mut omega_hat);
        self.dealias(&mut omega_hat);

        let mut u_hat = vec![Complex64::default(); n * n];
        let mut v_hat = vec![Complex64::default(); n * n];
        let mut dx_omega_hat = vec![Complex64::default(); n * n];
        let mut dy_omega_hat = vec![Complex64::default(); n * n];

        for l in 0..n {
            let y_k = ky[l];
            for j in 0..n {
                let x_k = kx[j];
                let k2 = x_k * x_k + y_k * y_k;
                let idx = l * n + j;
                let w = omega_hat[idx];

                if k2 > 0.0 {
                    let psi = w / k2;
                    u_hat[idx] = Complex64::new(0.0, y_k) * psi;
                    v_hat[idx] = Complex64::new(0.0, -x_k) * psi;
                }
                dx_omega_hat[idx] = Complex64::new(0.0, x_k) * w;
                dy_omega_hat[idx] = Complex64::new(0.0, y_k) * w;
            }
        }

        let mut u_c = vec![Complex64::default(); n * n];
        let mut v_c = vec![Complex64::default(); n * n];
        let mut dx_omega_c = vec![Complex64::default(); n * n];
        let mut dy_omega_c = vec![Complex64::default(); n * n];

        self.ifft2(&u_hat, &mut u_c);
        self.ifft2(&v_hat, &mut v_c);
        self.ifft2(&dx_omega_hat, &mut dx_omega_c);
        self.ifft2(&dy_omega_hat, &mut dy_omega_c);

        let mut prod_c = vec![Complex64::default(); n * n];
        for i in 0..n * n {
            prod_c[i] = Complex64::new(u_c[i].re * dx_omega_c[i].re + v_c[i].re * dy_omega_c[i].re, 0.0);
        }

        let mut prod_hat = vec![Complex64::default(); n * n];
        self.fft2(&prod_c, &mut prod_hat);
        self.dealias(&mut prod_hat);

        let mut rate_hat = vec![Complex64::default(); n * n];
        for l in 0..n {
            let y_k = ky[l];
            for j in 0..n {
                let x_k = kx[j];
                let k2 = x_k * x_k + y_k * y_k;
                let idx = l * n + j;
                rate_hat[idx] = -prod_hat[idx] - nu * k2 * omega_hat[idx];
            }
        }

        let mut rate_c = vec![Complex64::default(); n * n];
        self.ifft2(&rate_hat, &mut rate_c);
        for i in 0..n * n {
            out_rate[i] = rate_c[i].re;
        }
    }

    pub fn velocity_from_omega(&self, omega_real: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let (kx, ky) = self.wavenumbers();
        let omega_c = self.real_to_complex(omega_real);
        let mut omega_hat = vec![Complex64::default(); n * n];
        self.fft2(&omega_c, &mut omega_hat);
        self.dealias(&mut omega_hat);

        let mut u_hat = vec![Complex64::default(); n * n];
        let mut v_hat = vec![Complex64::default(); n * n];

        for l in 0..n {
            let y_k = ky[l];
            for j in 0..n {
                let x_k = kx[j];
                let k2 = x_k * x_k + y_k * y_k;
                let idx = l * n + j;
                if k2 > 0.0 {
                    let psi = omega_hat[idx] / k2;
                    u_hat[idx] = Complex64::new(0.0, y_k) * psi;
                    v_hat[idx] = Complex64::new(0.0, -x_k) * psi;
                }
            }
        }

        let mut u_c = vec![Complex64::default(); n * n];
        let mut v_c = vec![Complex64::default(); n * n];
        self.ifft2(&u_hat, &mut u_c);
        self.ifft2(&v_hat, &mut v_c);

        (self.complex_to_real(&u_c), self.complex_to_real(&v_c))
    }

    pub fn omega_from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        let n = self.n;
        let (kx, ky) = self.wavenumbers();

        let u_c = self.real_to_complex(u);
        let v_c = self.real_to_complex(v);

        let mut u_hat = vec![Complex64::default(); n * n];
        let mut v_hat = vec![Complex64::default(); n * n];
        self.fft2(&u_c, &mut u_hat);
        self.fft2(&v_c, &mut v_hat);

        let mut omega_hat = vec![Complex64::default(); n * n];
        for l in 0..n {
            let y_k = ky[l];
            for j in 0..n {
                let x_k = kx[j];
                let idx = l * n + j;
                omega_hat[idx] = Complex64::new(0.0, x_k) * v_hat[idx] - Complex64::new(0.0, y_k) * u_hat[idx];
            }
        }
        self.dealias(&mut omega_hat);

        let mut omega_c = vec![Complex64::default(); n * n];
        self.ifft2(&omega_hat, &mut omega_c);
        self.complex_to_real(&omega_c)
    }

    pub fn energy(&self, u: &[f64], v: &[f64]) -> f64 {
        let n = self.n;
        let sum_sq: f64 = u.iter().zip(v).map(|(&x, &y)| x * x + y * y).sum();
        0.5 * sum_sq / ((n * n) as f64)
    }

    pub fn enstrophy(&self, omega: &[f64]) -> f64 {
        let n = self.n;
        let sum_sq: f64 = omega.iter().map(|&w| w * w).sum();
        0.5 * sum_sq / ((n * n) as f64)
    }
}

pub fn taylor_green(n: usize, nu: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dx = 2.0 * PI / (n as f64);
    let mut u = vec![0.0; n * n];
    let mut v = vec![0.0; n * n];
    let mut omega = vec![0.0; n * n];
    let decay = (-2.0 * nu * t).exp();

    for l in 0..n {
        let y = (l as f64) * dx;
        for j in 0..n {
            let x = (j as f64) * dx;
            let idx = l * n + j;
            u[idx] = x.cos() * y.sin() * decay;
            v[idx] = -x.sin() * y.cos() * decay;
            omega[idx] = -2.0 * x.cos() * y.cos() * decay;
        }
    }
    (u, v, omega)
}

pub fn random_field(n: usize, seed: u64, k_min: f64, k_max: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let spec = Spectral2D::new(n);
    let k_max_int = k_max.ceil() as i32;
    let mut rng = Rng::seed_from_u64(seed);

    let mut omega_hat = vec![Complex64::default(); n * n];

    for ky in 0..=k_max_int {
        for kx in -k_max_int..=k_max_int {
            if ky > 0 || (ky == 0 && kx > 0) {
                let k = (((kx * kx + ky * ky) as f64)).sqrt();
                if k >= k_min && k <= k_max {
                    let phi = rng.gen_f64() * 2.0 * PI;
                    let val = Complex64::new(phi.cos(), phi.sin());

                    let ix = if kx >= 0 { kx as usize } else { (n as i32 + kx) as usize };
                    let iy = if ky >= 0 { ky as usize } else { (n as i32 + ky) as usize };

                    let ix_neg = if -kx >= 0 { (-kx) as usize } else { (n as i32 - kx) as usize };
                    let iy_neg = if -ky >= 0 { (-ky) as usize } else { (n as i32 - ky) as usize };

                    omega_hat[iy * n + ix] = val;
                    omega_hat[iy_neg * n + ix_neg] = val.conj();
                }
            }
        }
    }

    let (kx_arr, ky_arr) = spec.wavenumbers();
    let mut u_hat = vec![Complex64::default(); n * n];
    let mut v_hat = vec![Complex64::default(); n * n];

    for l in 0..n {
        let y_k = ky_arr[l];
        for j in 0..n {
            let x_k = kx_arr[j];
            let k2 = x_k * x_k + y_k * y_k;
            let idx = l * n + j;
            if k2 > 0.0 {
                let psi = omega_hat[idx] / k2;
                u_hat[idx] = Complex64::new(0.0, y_k) * psi;
                v_hat[idx] = Complex64::new(0.0, -x_k) * psi;
            }
        }
    }

    let mut u_c = vec![Complex64::default(); n * n];
    let mut v_c = vec![Complex64::default(); n * n];
    let mut omega_c = vec![Complex64::default(); n * n];

    spec.ifft2(&u_hat, &mut u_c);
    spec.ifft2(&v_hat, &mut v_c);
    spec.ifft2(&omega_hat, &mut omega_c);

    let mut u = spec.complex_to_real(&u_c);
    let mut v = spec.complex_to_real(&v_c);
    let mut omega = spec.complex_to_real(&omega_c);

    let e_raw = spec.energy(&u, &v);
    let scale = (0.5 / e_raw).sqrt();

    for i in 0..n * n {
        u[i] *= scale;
        v[i] *= scale;
        omega[i] *= scale;
    }

    (u, v, omega)
}
