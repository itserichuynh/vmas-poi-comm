"""
Train the Scouts & Catchers team (gnn_scenario.py) with MAPPO and a GNN policy, using BenchMARL.

    python train_gnn.py                          # full run (10M frames, ~35-40 min on a laptop GPU)
    python train_gnn.py --frames 60000           # quick test: one training iteration
    python train_gnn.py --n_scouts 0             # ablation: no scouts
    python train_gnn.py --comm_range 0.4         # ablation: shorter radio range (fewer graph edges)
    python train_gnn.py --moving_pois            # POIs drift around (speed: --poi_speed)
    python train_gnn.py --wandb                  # also log to Weights & Biases (run `wandb login` once first)
    python train_gnn.py --help                   # every setting

Each run gets a folder in ./results/ with:
    settings.json   - the settings used (watch_gnn.py reads this)
    checkpoints/    - saved models
    <run name>/scalars/ - CSV logs, e.g. episode reward over time

THE BIG PICTURE
    1. COLLECT   Run the policy in n_envs copies of the world for max_steps steps.
    2. SCORE     The critic estimates how good each moment was; compare that with what
                 actually happened -> "advantage" (better or worse than expected?).
    3. UPDATE    PPO nudges the policy toward actions with positive advantage.
    4. Repeat until `frames` steps have been collected.

THE NETWORKS
    Actor (policy) - what each robot runs to pick its action. Shared by all robots:
        node features (obs + pos + vel)
          -> GNN layer:  each robot mixes in information from robots within comm_range
          -> MLP:        turns that into an action (push x, push y)
    Critic - only used during training, to compute advantages. It's an MLP that sees
        every robot's observation at once ("centralized critic", the "MA" in MAPPO).
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch_geometric
from torchrl.envs import VmasEnv

from benchmarl.algorithms import MappoConfig
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import GnnConfig, MlpConfig, SequenceModelConfig

from gnn_scenario import Scenario

# ========================================================================== #
# SETTINGS: every one of these can be changed from the command line, e.g. --n_scouts 0
# ========================================================================== #
DEFAULTS = {
    # ---- Scenario (passed to gnn_scenario.py) ----
    "n_catchers": 4,               # robots that can catch POIs
    "n_scouts": 1,                 # robots that see far but can't catch
    "catcher_sensing_range": 0.15, # catchers are nearly blind
    "scout_sensing_range": 0.6,
    "comm_range": 1.0,             # robots closer than this are connected in the GNN graph
    "catcher_speed": 0.5,          # speed multiplier: 1.0 = normal (~0.4/s), 0.5 = half as fast
    "scout_speed": 1.0,
    "n_pois": 5,
    "n_poi_slots": 3,              # POIs each robot can report in its observation
    "catchers_needed": 2,          # catchers on a POI at the same time to catch it
    "moving_pois": False,          # --moving_pois: POIs drift around and bounce off the walls
    "poi_speed": 0.05,             # how fast moving POIs drift (a catcher at speed 1.0 does ~0.4)
    "closer_reward_weight": 1.0,   # "getting closer" bonus; 0 turns it off
    "closer_reward_mode": "gap",   # "gap": only the catcher completing each catch; "each": every catcher
    "max_steps": 200,              # steps per episode
    # ---- Training ----
    "frames": 10_000_000,          # total steps to train for (all copies together)
    "n_envs": 300,                 # copies of the world simulated in parallel
    "minibatch_iters": 45,         # gradient steps per training iteration
    "minibatch_size": 4096,        # steps per gradient step
    "lr": 5e-5,                    # learning rate
    "gnn_hidden": 128,             # size of the vector each robot computes in the GNN layer
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # ---- Logging ----
    "wandb": False,                # True: also log to Weights & Biases (the CSV logs are always saved)
    "wandb_project": "scouts-catchers",
}
SCENARIO_KEYS = [
    "n_catchers", "n_scouts", "catcher_sensing_range", "scout_sensing_range", "comm_range",
    "catcher_speed", "scout_speed",
    "n_pois", "n_poi_slots", "catchers_needed", "moving_pois", "poi_speed", "closer_reward_weight", "closer_reward_mode", "max_steps",
]


# ========================================================================== #
# STEP 1: tell BenchMARL how to build our environment
# ========================================================================== #
class ScoutsCatchersTask(VmasClass):
    """
    BenchMARL normally only knows VMAS's built-in scenarios. VmasClass already handles
    everything else (observation/action specs, groups, ...); we just swap in our Scenario.
    """

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        config = copy.deepcopy(self.config)  # the scenario settings + max_steps
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


# ========================================================================== #
# STEP 2: the policy network (actor) = GNN layer followed by an MLP
# ========================================================================== #
def make_actor_model(settings):
    gnn = GnnConfig(
        # --- Who is connected to whom ---
        topology="from_pos",             # build edges from positions every step...
        edge_radius=settings["comm_range"],  # ...between robots closer than this
        self_loops=False,                # (a robot already has its own features; no edge to itself)
        position_key="pos",              # the obs key holding positions
        pos_features=2,
        exclude_pos_from_node_features=False,  # also give each robot its own position (the arena has walls)
        velocity_key="vel",              # the obs key holding velocities
        vel_features=2,
        # --- What a GNN layer does ---
        # GATv2 = graph attention: each robot gives every neighbor an attention score
        # (computed from both robots' features and the edge features: relative position,
        # distance and relative velocity), then takes a weighted average of their vectors.
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={"add_self_loops": False, "residual": True},  # residual: also keep my own vector
    )
    mlp = MlpConfig(num_cells=[256, 256], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear)

    # Chain them: node features -> GNN -> (gnn_hidden numbers per robot) -> MLP -> action
    return SequenceModelConfig(model_configs=[gnn, mlp], intermediate_sizes=[settings["gnn_hidden"]])


def make_critic_model():
    # MAPPO's critic sees all robots' observations concatenated together.
    # A plain MLP is fine here: it's only used during training, never by the robots.
    return MlpConfig(num_cells=[256, 256], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear)


# ========================================================================== #
# STEP 3: the learning algorithm (MAPPO)
# ========================================================================== #
def make_algorithm():
    return MappoConfig(
        share_param_critic=True,        # one critic network for the whole team
        clip_epsilon=0.2,               # PPO: don't change the policy too much in one update
        entropy_coef=0.001,             # small bonus for staying random -> keeps exploring
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.9,                      # GAE lambda: how far ahead advantages look
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,           # actions are sampled from a squashed Gaussian in [-1, 1]
        minibatch_advantage=False,
    )


# ========================================================================== #
# STEP 4: put it all together
# ========================================================================== #
def build_experiment(settings, restore_file=None, watch_episodes=None):
    """
    Builds the BenchMARL Experiment from a settings dict.
    train_gnn.py calls it to train. watch_gnn.py calls it with restore_file=<checkpoint>
    and watch_episodes=N to rebuild the same networks and load trained weights.
    """
    task = ScoutsCatchersTask(name="SCOUTS_CATCHERS", config={k: settings[k] for k in SCENARIO_KEYS})

    config = ExperimentConfig.get_from_yaml()  # BenchMARL defaults, then our changes:
    config.sampling_device = config.train_device = settings["device"]
    config.max_n_frames = settings["frames"]
    config.gamma = 0.99                      # how much future rewards count
    config.lr = settings["lr"]
    config.share_policy_params = True        # one actor for every robot (role flag tells them apart)

    # One training iteration = n_envs full episodes
    frames_per_iteration = settings["n_envs"] * settings["max_steps"]
    config.on_policy_n_envs_per_worker = settings["n_envs"]
    config.on_policy_collected_frames_per_batch = frames_per_iteration
    config.on_policy_n_minibatch_iters = settings["minibatch_iters"]
    config.on_policy_minibatch_size = min(settings["minibatch_size"], frames_per_iteration)

    # Evaluate (deterministic actions, no learning) every 5 iterations; save a checkpoint every 20
    config.evaluation = True
    config.evaluation_interval = frames_per_iteration * 5
    config.evaluation_episodes = 50
    config.checkpoint_interval = frames_per_iteration * 20
    config.checkpoint_at_end = True
    config.render = False
    config.loggers = ["csv", "wandb"] if settings["wandb"] else ["csv"]
    config.project_name = settings["wandb_project"]  # BenchMARL uploads all settings to wandb itself
    config.save_folder = str(Path(__file__).parent / "results")
    Path(config.save_folder).mkdir(exist_ok=True)

    if restore_file is not None:  # used by watch_gnn.py
        import tempfile

        config.restore_file = str(restore_file)
        config.restore_map_location = settings["device"]
        config.save_folder = tempfile.mkdtemp()  # BenchMARL writes a few files on load; keep them out of results/
        config.loggers = []
        if watch_episodes is not None:
            config.evaluation_episodes = watch_episodes

    return Experiment(
        task=task,
        algorithm_config=make_algorithm(),
        model_config=make_actor_model(settings),
        critic_model_config=make_critic_model(),
        seed=settings["seed"],
        config=config,
    )


def main():
    # Turn every entry in DEFAULTS into a command-line option (--n_scouts 0, --frames 60000, ...)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in DEFAULTS.items():
        if isinstance(default, bool):  # on/off switches: --wandb turns it on
            parser.add_argument(f"--{name}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{name}", type=type(default), default=default)
    settings = vars(parser.parse_args())

    experiment = build_experiment(settings)
    with open(Path(experiment.folder_name) / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    experiment.run()
    print("\nDone. Results in:", experiment.folder_name)
    print("Watch it with:     python watch_gnn.py")


# ========================================================================== #
# Compatibility fixes for the installed library versions (you can ignore these)
# ========================================================================== #
# 1) BenchMARL's GNN builds its graph with torch_geometric's radius_graph, which needs the optional
#    pyg-lib / torch-cluster packages. They don't have builds for every torch version, so if they're
#    missing we use this plain-PyTorch version. The graphs are tiny (one node per robot), so it's fast.
def _radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target", **kwargs):
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    n_graphs = int(batch.max()) + 1
    n_nodes = x.shape[0] // n_graphs           # BenchMARL stacks one equal-size graph per world copy
    pos = x.view(n_graphs, n_nodes, -1)
    connected = torch.cdist(pos, pos) <= r     # [graphs, nodes, nodes]
    if not loop:
        connected &= ~torch.eye(n_nodes, dtype=torch.bool, device=x.device)
    graph, target, source = connected.nonzero(as_tuple=True)
    offset = graph * n_nodes                   # node numbers are global across the stacked graphs
    return torch.stack([source + offset, target + offset])


try:
    torch_geometric.nn.pool.radius_graph(torch.zeros(2, 2), r=1.0)
except ImportError:
    torch_geometric.nn.pool.radius_graph = _radius_graph

# 2) torchvision >= 0.29 removed io.write_video, which torchrl uses to save evaluation videos.
import torchvision.io

if not hasattr(torchvision.io, "write_video"):
    import imageio

    def _write_video(filename, video_array, fps, **kwargs):
        imageio.mimwrite(filename, torch.as_tensor(video_array).to(torch.uint8).cpu().numpy(), fps=float(fps), macro_block_size=1)

    torchvision.io.write_video = _write_video


if __name__ == "__main__":
    main()
