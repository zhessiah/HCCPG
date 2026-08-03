# SCCPG

A compact PyTorch implementation of **Self-Centered Clipped Critic Policy Gradient (SCCPG)** for decentralized continuous-control multi-agent reinforcement learning.

The implementation follows the paper's critic-to-actor mechanism:

1. Every agent fits an ordinary scalar critic.
2. Critic coefficient vectors are exchanged only inside a fixed local communication neighborhood.
3. Receiver `i` reconstructs sender `j`'s vector as

   ```text
   w_bar[j->i] = w_i + clip_tau(v[j->i] - w_i)
   ```

4. Reconstructed critics are evaluated and combined with nonnegative normalized local weights.
5. Only the receiver's actor is updated; actor parameters are never communicated.

This repository contains the core method and a single training interface. It does not include baselines, attacks, experiment reproduction, plotting, notebooks, or a test suite.

## Repository structure

```text
.
├── train.py
├── HCCPG/
│   ├── agent.py
│   ├── algorithm.py
│   ├── buffer.py
│   ├── clipping.py
│   ├── config.py
│   ├── envs.py
│   ├── networks.py
│   └── topology.py
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training

### CACC Catch-up

```bash
python train.py --env cacc-catchup --steps 1000000
```

### CACC Slowdown

```bash
python train.py --env cacc-slowdown --steps 1000000
```

### VMAS Dropout

```bash
python train.py --env vmas-dropout --steps 1000000 --num-envs 32
```

### VMAS Dispersion

```bash
python train.py --env vmas-dispersion --steps 1000000 --num-envs 32
```

Save a checkpoint:

```bash
python train.py --env cacc-catchup --steps 100000 --checkpoint checkpoints/cacc.pt
```

## Paper hyperparameter defaults

| Parameter | Default |
|---|---:|
| Actor learning rate | `5e-4` |
| Critic learning rate | `2.5e-4` |
| Optimizer | Adam |
| Discount factor | `0.99` |
| Actor minibatch size | `60` |
| Episode horizon | `100` |
| Actor hidden layers | `[64, 64]` |
| Critic hidden layers | `[64, 64]` |
| Activation | ReLU |
| Maximum gradient norm | `3.0` |
| Entropy coefficient | `0.1` |
| SCCPG clipping radius | `1.0` |
| Local neighborhood size | `3` |

All values can be overridden from `train.py`; run `python train.py --help` for the complete interface.

## Environment interfaces

The built-in CACC interface uses a longitudinal platoon model with a 0.1 s control interval, a 20 m target gap, bounded follower acceleration, a line communication graph, and a shared reward over spacing error, relative velocity, control effort, jerk, and collisions.

VMAS tasks use the official unwrapped `vmas.make_env` tensor API. `vmas-dropout` selects scenario `dropout`; `vmas-dispersion` selects scenario `dispersion`. Both use continuous actions and the configured episode horizon.

## Implementation notes

- The scalar critic receives the flattened joint observation and joint action and is trained against each agent's discounted local return.
- Incoming critic vectors are radially projected into the Euclidean ball centered at the receiver critic immediately before critic evaluation.
- The actor score-function objective retains the paper's outer factor `1 / (1 - gamma)`.
- Actor and critic parameters are projected onto a compact convex box after optimizer steps by elementwise clamping.
- CACC uses nearest-neighbor line aggregation; VMAS uses nearest-neighbor ring aggregation. Weights are uniform over each receiver's local neighborhood.

## Citation

```bibtex
@inproceedings{sccpg2026,
  title     = {Byzantine-Resilient Localized Policy Optimization with Self-Centered Parameter Clipping},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the 8th International Conference on Distributed Artificial Intelligence},
  year      = {2026}
}
```

## License

MIT
