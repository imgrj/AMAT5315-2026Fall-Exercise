import json
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spins_path = os.path.join(base_dir, "spins.jsonl")
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    target_temps = {1.8: "viewer-T1.8.png", 2.3: "viewer-T2.3.png", 3.0: "viewer-T3.0.png"}
    frames = {}

    with open(spins_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            t = round(obj['T'], 2)
            if t in target_temps:
                # keep the latest frame for this temperature (equilibrated)
                frames[t] = obj

    for t, out_name in target_temps.items():
        if t not in frames:
            print(f"Warning: T={t} not found in spins.jsonl")
            continue
        frame = frames[t]
        L = frame['L']
        spins = np.array(frame['spins']).reshape((L, L))
        m_val = frame['m']

        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
        # White (+1) and dark/black (-1)
        ax.imshow(spins, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f"T = {t:.1f},  M = {m_val:.4f}", fontsize=12, pad=8)
        ax.axis('off')
        plt.tight_layout()
        out_path = os.path.join(evidence_dir, out_name)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
