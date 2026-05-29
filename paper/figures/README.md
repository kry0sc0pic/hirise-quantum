# Figures

Figures to produce before submission (all referenced as PDF/EPS in main.tex):

| File | Source | How to generate |
|---|---|---|
| `architecture.pdf` | System diagram | Draw in Inkscape/TikZ: CNN encoder → shared weights → two branches → quantum head → similarity score |
| `circuit.pdf` | Quantum circuit | Use `qml.draw_mpl(circuit)` in PennyLane or draw in Quantikz LaTeX package |
| `confusion.pdf` | Eval results | `python -m eval.eval --checkpoint checkpoints/best_quantum.pt` → `eval_outputs/confusion_hard_pairs.png` |
| `gradnorm.pdf` | Training log | Loaded from checkpoint's `q_grad_history` field → `eval_outputs/grad_norms.png` |
| `umap.pdf` | Eval results | `eval_outputs/umap_quantum.png` (run eval.py after training) |

All `\res{X.XX}` placeholders in main.tex must be replaced with actual
numbers after running training + eval.
