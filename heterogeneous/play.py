"""Watch a trained policy play the POI capture scenario.

Examples:
    # newest checkpoint in ./results
    python play.py

    # a specific checkpoint
    python play.py --checkpoint results/<run>/checkpoints/checkpoint_10020000.pt

    # compare against random actions (same envs, same seed)
    python play.py --random

Saves a GIF of env 0 and prints catch stats averaged over --n_envs episodes.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment import Experiment

# config.pkl stores PoiTask as __main__.PoiTask (train.py was run as a script),
# so it has to be importable from this script's namespace for unpickling.
from train import PoiTask  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="path to checkpoint_*.pt (default: newest in ./results)")
    p.add_argument("--n_envs", type=int, default=32, help="parallel episodes to average stats over")
    p.add_argument("--steps", type=int, default=None, help="episode length (default: max_steps used in training)")
    p.add_argument("--out", default=None, help="GIF path (default: next to the checkpoint)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--random", action="store_true", help="ignore the policy, use random actions")
    p.add_argument("--device", default="cpu")
    p.add_argument("--poi_speed", type=float, default=None,
                   help="override POI speed (0 = static POIs; default: what the run was trained with)")
    p.add_argument("--comm_range", type=float, default=None)
    return p.parse_args()


def newest_checkpoint() -> Path:
    ckpts = sorted(Path(__file__).parent.glob("results/*/checkpoints/*.pt"), key=lambda f: f.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in ./results — train first or pass --checkpoint")
    return ckpts[-1]


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint()
    print("Checkpoint:", ckpt)

    # Rebuild the experiment exactly as trained and load the weights.
    # The patch just stops it from logging, rendering or using the GPU.
    # BenchMARL rewrites config.pkl on reload, so keep the original and put it back afterwards.
    config_pkl = ckpt.parent.parent / "config.pkl"
    original_config = config_pkl.read_bytes()

    # Override scenario settings without retraining (config.pkl only stores what training used)
    overrides = {}
    if args.poi_speed is not None:
        overrides["poi_speed"] = args.poi_speed
    if args.comm_range is not None:
        overrides['comm_range'] = args.comm_range
    if overrides:
        original_get_env_fun = PoiTask.get_env_fun

        def get_env_fun_with_overrides(self, *a, **kw):
            self.config.update(overrides)
            return original_get_env_fun(self, *a, **kw)

        PoiTask.get_env_fun = get_env_fun_with_overrides
        print("Scenario overrides:", overrides)

    experiment = Experiment.reload_from_file(
        str(ckpt),
        experiment_patch={
            "loggers": [],
            "render": False,
            "evaluation_episodes": args.n_envs,
            "sampling_device": args.device,
            "train_device": args.device,
            "restore_map_location": args.device,
        },
    )
    config_pkl.write_bytes(original_config)
    env = experiment.test_env  # n_envs parallel copies of the scenario
    policy = None if args.random else experiment.policy
    steps = args.steps or experiment.max_steps

    frames = []

    def grab_frame(env, td):
        frames.append(Image.fromarray(env.render(mode="rgb_array")))  # renders env 0

    exploration = ExplorationType.RANDOM if args.stochastic else ExplorationType.DETERMINISTIC
    with torch.no_grad(), set_exploration_type(exploration):
        rollout = env.rollout(
            max_steps=steps,
            policy=policy,
            callback=grab_frame,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )

    # Stats. Catches/reward are team-level, so read them from agent 0.
    group = next(iter(experiment.group_map))
    catches = rollout["next", group, "info", "catches"][..., 0].sum(dim=1)  # [n_envs]
    reward = rollout["next", group, "reward"][..., 0, 0].sum(dim=1)  # [n_envs]
    print(f"Policy:  {'random' if args.random else 'trained'} ({'stochastic' if args.stochastic else 'deterministic'})")
    print(f"Episodes: {args.n_envs} x {steps} steps")
    print(f"POIs caught per episode: {catches.mean():.2f} ± {catches.std():.2f} (min {catches.min():.0f}, max {catches.max():.0f})")
    print(f"Team return per episode: {reward.mean():.2f}")

    out = Path(args.out) if args.out else ckpt.parent.parent / ("play_random.gif" if args.random else "play.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0)
    print("GIF saved to:", out.resolve())


if __name__ == "__main__":
    main()
