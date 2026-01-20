## 🔗 Related Technical Article

This repository is accompanied by the following public technical article, which explains the motivation, methodology, and interpretation of autoencoder-based residual analysis for TRNG monitoring:

- **AI-Assisted TRNG Entropy Analysis Using SP800-90B and Autoencoder Residual Mapping**  
  https://medium.com/@ace.lin0121/ai-assisted-trng-entropy-analysis-using-sp800-90b-and-autoencoder-residual-mapping-cdf2ca6e3cb1

The article provides architectural context and visualization insights that complement this runnable, synthetic demo.


# TRNG AE Monitor (Synthetic Demo)

> **Version 2 – Code-only Release (Jan 2026)**  
> This release provides a clean, runnable **code-only** reference implementation.  
> Datasets (`*.npz`) and trained checkpoints (`*.pt`) are intentionally excluded and must be generated locally.

This is a runnable PyTorch demo of a CNN autoencoder (AE) that monitors TRNG-like bit-plane windows.

- Input: P×64×64 bit-planes (float {0,1})
- Train AE on **healthy** samples only
- Inject several **degraded/error patterns** for evaluation
- Output per-frame MSE score + per-bit-plane MSE + simple signature tags

## 1) Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Generate training/validation data (healthy only)

```bash
python gen_data.py --out data/train_healthy.npz --n_healthy 8000 --n_degraded 0 --seed 1
python gen_data.py --out data/val_healthy.npz   --n_healthy 2000 --n_degraded 0 --seed 2
```

## 3) Train the autoencoder

```bash
python train.py --train_npz data/train_healthy.npz --val_npz data/val_healthy.npz --epochs 25 --batch 64 --out_dir checkpoints
```

This writes:
- `checkpoints/best.pt` (model + q99/q99.9 thresholds from validation)
- `checkpoints/train_history.json`

## 4) Generate an evaluation dataset with injected error patterns

```bash
python gen_data.py --out data/eval_mix.npz --n_healthy 2000 --n_degraded 2000 --seed 3
```

Degraded patterns included (mix):
- bias
- stuck_plane
- block_stuck
- stripes
- burst

## 5) Evaluate and write a JSON report

```bash
python eval.py --npz data/eval_mix.npz --ckpt checkpoints/best.pt --out_json outputs/eval_report.json
```

The report contains:
- thresholds (q99, q99.9)
- score stats for healthy vs degraded
- first 200 detailed records (label, inj_type, score, per-plane MSE, warn/fail flags, signature tags)

## Notes

- This is a **synthetic** demo to provide a safe, reproducible framework.
- It does **not** claim to model hardware root causes.

## Scope, Baselines, and Reproducibility Notes

This repository provides a **code-only reference implementation** accompanying the manuscript
“Continuous Entropy Monitoring of TRNGs via a CNN Autoencoder with Residual-Based Diagnostics”.

**Baselines.**  
The evaluation focuses on engineering-relevant baselines rather than comparisons with alternative
machine learning classifiers. Statistical estimators aligned with NIST SP 800-90B are treated as
the compliance ground truth for entropy quality, while the autoencoder is used strictly as an
anomaly detector for deployment-time monitoring.

**Generality.**  
The monitoring framework is agnostic to the specific number of bit-planes \(P\).
While the canonical framing supports \(P = 8\), the evaluation dataset used in the manuscript
provides \(P = 6\). All training and evaluation are therefore performed using a 6-plane variant
to ensure a consistent input space across experiments. The methodology does not rely on
TRNG-specific physical models and can be applied to other bitstream sources with the same framing.

**Reproducibility.**  
Example output files under `example/` are provided for illustration purposes only.
Reported figures and score trends in the manuscript can be reproduced by running the scripts on
synthetic or publicly shareable bitstreams, as described in the paper.
No proprietary or confidential device data are included.


## 6) Generate figures (PNG)

```bash
python plot_results.py --npz data/eval_mix.npz --ckpt checkpoints/best.pt --out_dir outputs/figures
```

Outputs:
- `outputs/figures/fig_score_hist.png` (Figure: score histogram)
- `outputs/figures/fig_plane_mse.png` (Figure: per-plane MSE bar chart)
- `outputs/figures/fig_residual_XX.png` (Residual-map examples)
