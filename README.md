# simclr-pytorch

Compact implementation of SimCLR in PyTorch.
The implementation uses a ResNet-18 encoder, a two-layer projection head, and NT-Xent loss.

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 🧪 Pretraining

```bash
python src/pretrain.py --epochs 200 --batch-size 256 --lr 3e-4
```

## 📈 Linear Evaluation

```bash
python src/linear_eval.py --checkpoint checkpoints/simclr_latest.pt --epochs 50 --batch-size 256 --lr 1e-2
```

## 📁 Structure

```text
src/
  augmentations.py
  model.py
  loss.py
  pretrain.py
  linear_eval.py
  utils.py
```
