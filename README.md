# Multi-Robot POI Capture

A team of robots learns to catch points of interest (POIs) together.
A POI only counts as caught when 2 robots are on it at the same time, so the robots have to work as a team.
Each robot can only see things close to it, but robots can send each other short messages if within range to share what they've found.

## Files

| File | What it does |
|---|---|
| `poi_scenario.py` | The world: robots, POIs, rewards and what each robot sees |
| `train.py` | Trains the robots |
| `play.py` | Loads a trained model and makes a GIF of the robots playing |
| `scenario_setup.ipynb` | Notebook that explains how the world was built, step by step |
| `results/` | Where training saves logs, models and videos (created automatically) |

---

## Step 1: Set up (one time only)

You need [Anaconda or Miniconda](https://www.anaconda.com/download). Open a terminal and run:

```bash
conda create -n marl python=3.10 -y
conda activate marl
pip install torch torchvision benchmarl==1.5.2 vmas==1.5.2 imageio imageio-ffmpeg pyglet==1.5.27 wandb
```

> Every time you open a new terminal, run `conda activate marl` first.

## Step 2: Check that the world works

```bash
python poi_scenario.py
```

You should see something like:

```
Robots: 4
Obs shape (robot 0): (8, 37)
Action space (robot 0): Box([-1. -1.  0.  0.  0.  0.], 1.0, (6,), float32)
Random policy catches per env over 200 steps: [0.0, 0.0, ...]
```

Robots that move randomly in the output gif.

---

## Step 3: Train

Quick test if you want to.

```bash
python train.py --frames 6000 --n_envs 30 --minibatch_iters 1
```

Then real training with MAPPO

```bash
python train.py
```

- This runs 10 million steps. It uses the GPU automatically if there is one. Didn't take me long on my computer.
- You'll see a progress bar with `mean return = ...`.
- When it finishes, it prints where it saved everything, e.g. `results/mappo_poi_capture_mlp__xxxx/`.

**Optional extras args:**
- Add `--render` to also save a video every few evaluations.
- Add `--wandb` to see live graphs on [wandb.ai](https://wandb.ai). Run `wandb login` once first.

---

## Step 4: Watch the trained robots

To use a specific trained model:

```bash
python play.py --checkpoint results/<run name>/checkpoints/checkpoint_10020000.pt
```

---

## How the robots are rewarded

The whole team shares one score. Each step:

- **+1** for every POI caught. The POI then jumps to a new random spot.
- **A small bonus** for getting closer to POIs (and a small penalty for moving away).
- **-0.01** every step, so being faster is better.
- **-0.1** for a robot that bumps into a teammate.

---

## Where the results are

Each training run gets its own folder in `results/`:

```
results/mappo_poi_capture_mlp__xxxx/
├── checkpoints/checkpoint_10020000.pt   ← the trained model
├── config.pkl                           ← settings used (play.py reads this)
├── play.gif                             ← made by play.py
└── mappo_poi_capture_mlp__xxxx/
    ├── scalars/                         ← CSV logs (rewards over time, etc.)
    └── videos/                          ← only if you trained with --render
```

