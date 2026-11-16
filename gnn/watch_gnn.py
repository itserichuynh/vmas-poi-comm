"""
Watch a trained Scouts & Catchers team play.

    python watch_gnn.py                       # newest checkpoint in ./results
    python watch_gnn.py --checkpoint results/<run>/checkpoints/checkpoint_10020000.pt
    python watch_gnn.py --random              # random actions, for comparison

Change the scenario the trained team plays in (no retraining), e.g. a team trained on
still-standing POIs, now playing with moving ones:
    python watch_gnn.py --checkpoint <...> --moving_pois --poi_speed 0.1

    Any scenario setting works: --scout_sensing_range 0.3, --comm_range 0.5, --n_pois 8, ...
    Except: n_catchers, n_scouts and n_poi_slots change the network's input sizes, so the
    saved weights won't fit.

Runs `--episodes` episodes in parallel, prints how many POIs the team caught,
and saves a GIF of the first episode next to the checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchrl.envs.utils import ExplorationType, set_exploration_type

from train_gnn import DEFAULTS, SCENARIO_KEYS, build_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="default: newest checkpoint in ./results")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--random", action="store_true", help="ignore the trained policy and act randomly")
parser.add_argument("--fps", type=int, default=20)
# One option per scenario setting. Left out (None) = keep what the run was trained with.
for name in SCENARIO_KEYS:
    if isinstance(DEFAULTS[name], bool):
        parser.add_argument(f"--{name}", action="store_const", const=True, default=None)
    else:
        parser.add_argument(f"--{name}", type=type(DEFAULTS[name]), default=None)
args = parser.parse_args()

# ---- 1. Find the checkpoint and the settings it was trained with ----
if args.checkpoint:
    checkpoint = Path(args.checkpoint)
else:
    all_checkpoints = Path(__file__).parent.glob("results/*/checkpoints/*.pt")
    checkpoint = max(all_checkpoints, key=lambda f: f.stat().st_mtime)  # newest
run_folder = checkpoint.parent.parent
settings = json.loads((run_folder / "settings.json").read_text())
# Runs saved before a setting existed don't have it in settings.json: fill in its default
settings = {**DEFAULTS, **settings}
settings["device"] = "cpu"  # watching is cheap; no GPU needed
changes = {k: getattr(args, k) for k in SCENARIO_KEYS if getattr(args, k) is not None}
settings.update(changes)
print("Checkpoint:", checkpoint)
if changes:
    print("Changed from training:", changes)

# ---- 2. Rebuild the same networks and load the trained weights ----
experiment = build_experiment(settings, restore_file=checkpoint, watch_episodes=args.episodes)
env = experiment.test_env          # `episodes` copies of the world
policy = None if args.random else experiment.policy  # policy=None -> random actions

# ---- 3. Play one episode in every copy, grabbing a frame of copy 0 each step ----
frames = []
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):  # best action, no sampling
    rollout = env.rollout(
        max_steps=settings["max_steps"],
        policy=policy,
        callback=lambda env, _: frames.append(Image.fromarray(env.render(mode="rgb_array"))),
        auto_cast_to_device=True,
        break_when_any_done=False,
    )

# ---- 4. Results ----
# rollout["next", "agents", "info", "catches"] has shape [copies, steps, robots, ...].
# Catches are team-wide (the same for every robot), so read them from robot 0.
catches = rollout["next", "agents", "info", "catches"][:, :, 0].reshape(args.episodes, -1).sum(dim=1)
print(f"Policy: {'random' if args.random else 'trained'}")
print(f"POIs caught per episode: {catches.mean():.2f} ± {catches.std():.2f}  (min {catches.min():.0f}, max {catches.max():.0f})")

# e.g. watch.gif, watch_random.gif, watch_moving_poisTrue_poi_speed0.1.gif
gif_name = "_".join(["watch"] + (["random"] if args.random else []) + [f"{k}{v}" for k, v in changes.items()])
gif = run_folder / (gif_name + ".gif")
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0)
print("GIF saved to:", gif.resolve())
