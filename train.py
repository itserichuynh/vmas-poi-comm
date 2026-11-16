"""Train agents on the POI capture scenario with BenchMARL.

Examples:
    # quick smoke test (one iteration)
    python train.py --frames 6000 --n_envs 30 --minibatch_iters 1

    # full MAPPO run with communication
    python train.py --algo mappo --frames 10_000_000

    # ablation: no communication
    python train.py --dim_c 0

    # ablation: lossy radio
    python train.py --comm_dropout 0.3 --comm_noise 0.1

    # GNN policy (needs torch_geometric)
    python train.py --model gnn

Results (CSV logs, checkpoints, eval videos) go to ./results/<run name>/.
"""

import argparse
import copy
from pathlib import Path
from typing import Callable, Optional

import torch
from torchrl.envs import EnvBase, VmasEnv

from benchmarl.algorithms import IppoConfig, MappoConfig, MasacConfig
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.utils import DEVICE_TYPING

from poi_scenario import Scenario

# torchvision >= 0.29 removed io.write_video, which torchrl's CSV logger uses to save eval videos.
# Put back a minimal version backed by imageio so --render works.
import torchvision.io

if not hasattr(torchvision.io, "write_video"):
    import imageio

    def _write_video(filename, video_array, fps, **kwargs):
        frames = torch.as_tensor(video_array).to(torch.uint8).cpu().numpy()  # [T, H, W, C]
        imageio.mimwrite(filename, frames, fps=float(fps), macro_block_size=1)

    torchvision.io.write_video = _write_video


class PoiTask(VmasClass):
    """BenchMARL task that builds our custom VMAS scenario instead of a built-in one."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: VmasEnv(
            scenario=Scenario(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )


def parse_args():
    p = argparse.ArgumentParser()
    # Scenario
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--n_pois", type=int, default=3)
    p.add_argument("--agents_per_poi", type=int, default=2)
    p.add_argument("--sensing_range", type=float, default=0.35)
    p.add_argument("--dim_c", type=int, default=4)
    p.add_argument("--comm_range", type=float, default=1.0)
    p.add_argument("--comm_dropout", type=float, default=0.0)
    p.add_argument("--comm_noise", type=float, default=0.0)
    p.add_argument("--closer_reward_weight", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=200)
    # Training
    p.add_argument("--algo", choices=["mappo", "ippo", "masac"], default="mappo")
    p.add_argument("--model", choices=["mlp", "gnn"], default="mlp")
    p.add_argument("--frames", type=int, default=10_000_000)
    # p.add_argument("--frames", type=int, default=10_000)
    p.add_argument("--n_envs", type=int, default=300, help="vectorized envs per collection")
    p.add_argument("--minibatch_iters", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render", action="store_true", help="save eval videos (needs a display / xvfb)")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def make_algorithm(name: str):
    if name == "mappo":
        return MappoConfig(
            share_param_critic=True,
            clip_epsilon=0.2,
            entropy_coef=0.001,
            critic_coef=1,
            loss_critic_type="l2",
            lmbda=0.9,
            scale_mapping="biased_softplus_1.0",
            use_tanh_normal=True,
            minibatch_advantage=False,
        )
    if name == "ippo":
        cfg = IppoConfig.get_from_yaml()
        cfg.entropy_coef = 0.001
        return cfg
    return MasacConfig.get_from_yaml()


def make_model(name: str, comm_range: float):
    if name == "mlp":
        return MlpConfig(num_cells=[256, 256], layer_class=torch.nn.Linear, activation_class=torch.nn.Tanh)

    import torch_geometric
    from benchmarl.models import GnnConfig, SequenceModelConfig

    gnn = GnnConfig(
        topology="from_pos",  # edges between agents within comm_range
        edge_radius=comm_range,
        self_loops=False,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={"add_self_loops": False, "residual": True},
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=True,
    )
    return SequenceModelConfig(model_configs=[gnn, MlpConfig.get_from_yaml()], intermediate_sizes=[256])


def main():
    args = parse_args()

    task = PoiTask(
        name="POI_CAPTURE",
        config={
            "max_steps": args.max_steps,
            "n_agents": args.n_agents,
            "n_pois": args.n_pois,
            "agents_per_poi": args.agents_per_poi,
            "sensing_range": args.sensing_range,
            "dim_c": args.dim_c,
            "comm_range": args.comm_range,
            "comm_dropout": args.comm_dropout,
            "comm_noise": args.comm_noise,
            "closer_reward_weight": args.closer_reward_weight,
        },
    )

    exp = ExperimentConfig.get_from_yaml()
    exp.sampling_device = args.device
    exp.train_device = args.device
    exp.max_n_frames = args.frames
    exp.gamma = 0.99
    exp.share_policy_params = True  # one policy shared by all robots

    # On-policy (MAPPO / IPPO): each iteration collects n_envs full episodes
    exp.on_policy_n_envs_per_worker = args.n_envs
    exp.on_policy_collected_frames_per_batch = args.n_envs * args.max_steps
    exp.on_policy_n_minibatch_iters = args.minibatch_iters
    exp.on_policy_minibatch_size = min(4096, exp.on_policy_collected_frames_per_batch)

    # Off-policy (MASAC)
    exp.off_policy_n_envs_per_worker = args.n_envs
    exp.off_policy_collected_frames_per_batch = args.n_envs * args.max_steps

    exp.evaluation = True
    exp.evaluation_interval = exp.on_policy_collected_frames_per_batch * 5
    exp.evaluation_episodes = 50
    exp.render = args.render
    exp.loggers = ["csv", "wandb"] if args.wandb else ["csv"]
    exp.save_folder = str(Path(__file__).parent / "results")
    Path(exp.save_folder).mkdir(exist_ok=True)
    exp.checkpoint_at_end = True

    experiment = Experiment(
        task=task,
        algorithm_config=make_algorithm(args.algo),
        model_config=make_model(args.model, args.comm_range),
        critic_model_config=MlpConfig(num_cells=[256, 256], layer_class=torch.nn.Linear, activation_class=torch.nn.Tanh),
        seed=args.seed,
        config=exp,
    )
    experiment.run()
    print("Done. Logs and checkpoint in:", experiment.folder_name)


if __name__ == "__main__":
    main()
