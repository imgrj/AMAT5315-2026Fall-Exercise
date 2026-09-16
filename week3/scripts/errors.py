import json
import os
import numpy as np

def compute_autocorr(x, max_lag=10000):
    n = len(x)
    mu = np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return np.ones(max_lag), 0.5

    max_lag = min(max_lag, n - 1)
    # FFT-based fast autocorrelation
    x_zero = x - mu
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x_zero, n=n_fft)
    r = np.fft.irfft(fx * np.conj(fx), n=n_fft)[:n]
    # normalise by degrees of freedom
    norm = np.arange(n, 0, -1)
    acf = (r / norm) / var
    acf = acf[:max_lag]

    # Integrated autocorrelation time: tau_int = 0.5 + sum_{t=1..} acf[t]
    # Truncate when t > 6 * tau_int
    tau_int = 0.5
    for t in range(1, len(acf)):
        tau_int += acf[t]
        if t > 6.0 * tau_int:
            break

    return acf, max(0.5, tau_int)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    errors_txt_path = os.path.join(evidence_dir, "errors.txt")

    results = []
    # Header
    results.append("L\tT\tmean |M|\tnaive se\t50-block se\terror ratio\ttau_int")

    for size in [32, 64]:
        coarse_path = os.path.join(base_dir, "artifacts", f"coarse-l{size}", "series.jsonl")
        window_path = os.path.join(base_dir, "artifacts", f"window-l{size}", "series.jsonl")

        t_dict = {}
        if os.path.exists(coarse_path):
            with open(coarse_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    t = round(float(d['T']), 4)
                    if t not in t_dict: t_dict[t] = []
                    t_dict[t].append(abs(float(d['M'])))

        if os.path.exists(window_path):
            w_dict = {}
            with open(window_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    t = round(float(d['T']), 4)
                    if t not in w_dict: w_dict[t] = []
                    w_dict[t].append(abs(float(d['M'])))
            for t, m_list in w_dict.items():
                t_dict[t] = m_list

        for t in sorted(t_dict.keys()):
            x = np.array(t_dict[t])
            n = len(x)
            mean_m = np.mean(x)
            naive_se = np.std(x, ddof=1) / np.sqrt(n)

            # 50 block standard error
            n_blocks = 50
            block_len = n // n_blocks
            x_blocks = x[:n_blocks * block_len].reshape((n_blocks, block_len))
            block_means = np.mean(x_blocks, axis=1)
            block_se = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

            ratio = block_se / naive_se if naive_se > 1e-12 else 1.0

            # tau_int
            _, tau_int = compute_autocorr(x, max_lag=min(n//2, 20000))

            results.append(f"{size}\t{t:.2f}\t{mean_m:.6f}\t{naive_se:.6f}\t{block_se:.6f}\t{ratio:.2f}\t{tau_int:.2f}")

    output_str = "\n".join(results) + "\n"
    with open(errors_txt_path, 'w') as f:
        f.write(output_str)
    print(f"Saved {errors_txt_path}")

    # Print check lines
    print("\nCheck three L = 64 rows (T = 1.5, 2.3, 3.5):")
    for line in results:
        parts = line.split('\t')
        if len(parts) >= 7 and parts[0] == '64' and parts[1] in ['1.50', '2.30', '3.50']:
            print(f"T={parts[1]}: error ratio = {parts[5]}, tau_int = {parts[6]} sweeps")

if __name__ == '__main__':
    main()
