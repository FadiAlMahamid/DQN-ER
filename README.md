# DQN (2015) with Experience Replay: Atari Breakout

A **Deep Q-Network (DQN)** implementation with **Experience Replay** and a **Target Network** applied to `BreakoutNoFrameskip-v4` from OpenAI Gymnasium. This project implements the algorithm from the DeepMind 2015 Nature paper: a **Replay Buffer** to break temporal correlations and a **Target Network** to solve the moving target problem.

## Project Overview

The agent learns to play Atari Breakout using a convolutional neural network to approximate Q-values. The implementation incorporates two key innovations from the 2015 paper:

1. **Experience Replay Buffer** — Stores transitions and samples random mini-batches for training, breaking the correlation between consecutive samples and preventing catastrophic forgetting.
2. **Target Network** — A periodically-updated frozen copy of the policy network provides stable TD targets, solving the moving target problem where the network chases its own changing predictions.

### How This Differs from Double DQN

The standard DQN target calculation uses the target network for **both** selecting and evaluating the best next action:

```
DQN (2015):     target = R + γ * max_a' Q_target(s', a')
Double DQN:     target = R + γ * Q_target(s', argmax_a' Q_policy(s', a'))
```

In standard DQN, the `max` operator can systematically **overestimate** Q-values because the same network both picks the best action and evaluates it. Double DQN (van Hasselt et al., 2016) addresses this by using the policy network to *select* the action and the target network to *evaluate* it. This decoupling reduces overestimation bias.

### Core Features

- **Modular Architecture:** Separated into distinct files for the environment, agent, network, replay buffer, utilities, and training logic.
- **Step-Based Training Loop:** Training runs for a fixed number of agent steps (not episodes), matching the original DeepMind approach. All logging, checkpointing, and epsilon decay are step-based.
- **YAML Configuration:** All hyperparameters, paths, and execution modes are controlled via `config.yaml` — no need to edit code between experiments.
- **GPU/MPS Support:** Automatic device detection for CUDA, Apple Silicon (MPS), or CPU fallback.
- **Persistent Storage:**
  - `.pth` files store the trained model weights.
  - `.npz` files store the full training history (rewards, losses, step count).
- **Resume Training:** Interrupt training at any time (Ctrl+C) and resume from the last checkpoint without losing progress.
- **Visualization:** Automatic generation of reward and loss training curves at each checkpoint, plus a deployment mode with action distribution logging.

---

## Project Structure

```
DQN-ER/
├── config.yaml            # All hyperparameters and settings
├── q_network.py           # CNN architecture (DeepMind 2015 style)
├── replay_buffer.py       # Memory-efficient Experience Replay Buffer
├── dqn_agent.py           # DQN agent (policy + target networks)
├── environment.py         # Gym environment setup with Atari preprocessing
├── utils.py               # Config loading, plotting, and deployment visualization
├── training_script.py     # Step-based training loop and main entry point
└── dqn_results/           # Generated outputs (model, history, plots)
```

---

## Key Components

| Component | Purpose |
|---|---|
| **Experience Replay Buffer** | Stores individual frames (uint8) and reconstructs stacked states on demand, breaking temporal correlation while minimizing memory usage |
| **Target Network** | A frozen copy of the policy network, updated every 10,000 steps, solves the moving target problem |
| **Dedicated Warmup** | `train()` calls `warmup()` internally to fill the replay buffer with random-action transitions before the training loop begins, avoiding per-step buffer-size checks during learning |
| **Periodic Checkpointing** | Saves model, history, and training plots every 100,000 steps to protect against crashes |

### DQN (2015) Target Calculation

```
target = R + γ * max_a' Q_target(s', a')
```

The target network provides a **stable** set of Q-values that don't change between updates. Without it, the policy network would be chasing a moving target — each gradient step changes the predictions, which changes the targets, creating a feedback loop that destabilizes training.

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Gymnasium with Atari support (`ale-py`)
- NumPy, Matplotlib, PyYAML

### Installation

```bash
pip install torch gymnasium ale-py numpy matplotlib pyyaml
```

### Running the Script

Control the execution mode via `config.yaml`:

```yaml
training:
  mode: "new"      # Train from scratch
  # mode: "resume" # Load saved model and continue training
  # mode: "deploy" # Watch the trained agent play
```

Then run:

```bash
python training_script.py
```

---

## Configuration

All parameters are managed in [`config.yaml`](config.yaml). The config is organized into logical sections:

### Training Loop

| Parameter | Default | Description |
|---|---|---|
| `num_steps` | 2,500,000 | Total agent steps to train for (each step = 4 frames) |
| `target_reward` | 40 | Average reward for early stopping (per-life, not per-game) |
| `print_every` | 10,000 | Steps between console progress logs |
| `checkpoint_every` | 100,000 | Steps between model/plot checkpoints (0 to disable) |
| `plot_window` | 500 | Moving average window for reward curves and early stopping |

### Agent Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | rmsprop | Optimizer type (`adam` or `rmsprop`) |
| `learning_rate` | 0.00025 | Optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `loss_function` | huber | Loss type (`huber` or `mse`) |
| `grad_clip_norm` | 1.0 | Maximum gradient norm for clipping |
| `clip_rewards` | true | Clip rewards to [-1, 1] for stable gradients |

### Epsilon-Greedy Exploration

| Parameter | Default | Description |
|---|---|---|
| `start` | 1.0 | Initial exploration rate (100% random) |
| `min` | 0.01 | Minimum exploration rate after decay completes |
| `decay_steps` | 1,000,000 | Steps for linear epsilon decay |

### Experience Replay Buffer

| Parameter | Default | Description |
|---|---|---|
| `capacity` | 1,000,000 | Maximum transitions stored (FIFO eviction) |
| `batch_size` | 32 | Mini-batch size for training |
| `learning_starts` | 50,000 | Warmup transitions before learning begins |

### Target Network

| Parameter | Default | Description |
|---|---|---|
| `update_freq` | 10,000 | Steps between target network weight copies |

### Other

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Random seed for reproducibility |

---

## References

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
