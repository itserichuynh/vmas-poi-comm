"""Cooperative POI capture scenario (VMAS).

Robots must gather on points of interest (POIs) in groups of `agents_per_poi`
to catch them.

Run this file directly to sanity-check the env and save a GIF of random actions:
    python poi_scenario.py
"""

import torch
from torch import Tensor

from vmas import make_env
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.environment import Environment
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

# VMAS prints "agent_i sends [...]" in the corner of every rendered frame when dim_c > 0.
# Set to True to bring that text back.
# SHOW_MESSAGE_TEXT = False
# if not SHOW_MESSAGE_TEXT:
#     Environment._set_agent_comm_messages = lambda self, env_index: None


class Scenario(BaseScenario):
    # ------------------------------------------------------------------ #
    # World construction
    # ------------------------------------------------------------------ #
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_catchers = kwargs.pop("n_catchers", 4)
        self.n_scouts = kwargs.pop("n_scouts", 1)
        self.n_agents = self.n_catchers + self.n_scouts
        self.catcher_sensing_range = kwargs.pop("catcher_sensing_range", 0.15)
        self.scout_sensing_range = kwargs.pop("scout_sensing_range", 0.6)

        self.n_pois = kwargs.pop("n_pois", 3)
        self.agents_per_poi = kwargs.pop("agents_per_poi", 2)
        self.arena_size = kwargs.pop("arena_size", 1.0)
        # self.sensing_range = kwargs.pop("sensing_range", 0.5)

        # self.dim_c = kwargs.pop("dim_c", 0)
        self.comm_range = kwargs.pop("comm_range", 1.0)
        # self.comm_dropout = kwargs.pop("comm_dropout", 0.0)
        # self.comm_noise = kwargs.pop("comm_noise", 0.0)

        self.capture_range = kwargs.pop("capture_range", 0.15)
        self.capture_reward = kwargs.pop("capture_reward", 1.0)
        self.closer_reward_weight = kwargs.pop("closer_reward_weight", 1.0)
        self.time_penalty = kwargs.pop("time_penalty", -0.01)
        self.collision_penalty = kwargs.pop("collision_penalty", -0.1)

        self.goal_vel = torch.zeros(batch_dim, self.n_pois, 2, device=device)
        self.poi_speed = kwargs.pop('poi_speed', 0.05)

        self.agent_radius = 0.05
        ScenarioUtils.check_kwargs_consumed(kwargs)  # complain about unknown settings (typos)

        world = World(
            batch_dim,
            device,
            # dim_c=self.dim_c,  # message size (0 = no messages)
            x_semidim=self.arena_size,  # invisible walls
            y_semidim=self.arena_size,
        )

        # Robots
        # for i in range(self.n_agents):
        #     robot = Agent(
        #         name=f"agent_{i}",
        #         shape=Sphere(radius=self.agent_radius),
        #         color=Color.BLUE,
        #         collide=True,
        #         # silent=(self.dim_c == 0),  # no message actions when messages are off
        #         # c_noise=self.comm_noise,  # fuzz added to messages
        #         u_multiplier=2.0,
        #     )
        #     robot.collision_reward = torch.zeros(batch_dim, device=device)  # filled in by reward()
        #     world.add_agent(robot)

        # Robots: catchers first (agent_0 ..), then scouts
        self.catchers, self.scouts = [], []
        for i in range(self.n_agents):
            is_scout = i >= self.n_catchers
            robot = Agent(
                name=f"agent_{i}",
                shape=Sphere(radius=self.agent_radius),
                color=Color.GREEN if is_scout else Color.BLUE,
                collide=not is_scout,  # scouts fly over everything
                u_multiplier=2.0,
            )
            robot.is_scout = is_scout
            robot.sensing_range = self.scout_sensing_range if is_scout else self.catcher_sensing_range
            robot.collision_reward = torch.zeros(batch_dim, device=device)  # filled in by reward()
            world.add_agent(robot)
            (self.scouts if is_scout else self.catchers).append(robot)

        # POIs
        self.pois = []
        for i in range(self.n_pois):
            poi = Landmark(
                name=f"poi_{i}",
                shape=Sphere(radius=0.03),
                color=Color.RED,
                collide=False,  # robots pass through POIs
                movable=True
            )
            world.add_landmark(poi)
            self.pois.append(poi)

        # Places to keep reward info between steps
        self.team_reward = torch.zeros(batch_dim, device=device)
        self.catches = torch.zeros(batch_dim, device=device)
        self.closer_reward = torch.zeros(batch_dim, device=device)
        self.previous_gap = torch.zeros(batch_dim, self.n_pois, device=device)

        return world

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _robot_poi_distances(self) -> Tensor:
        # Shape [env id, robots, POIs]
        # robot_positions = torch.stack([robot.state.pos for robot in self.world.agents], dim=1)
        robot_positions = torch.stack([robot.state.pos for robot in self.catchers], dim=1)
        poi_positions = torch.stack([poi.state.pos for poi in self.pois], dim=1)
        return torch.cdist(robot_positions, poi_positions)

    def _gap_to_capture(self) -> Tensor:
        # For each POI, the distance of the k-th closest robot (k = agents_per_poi). Shape [env ids, POIs]
        sorted_distances = self._robot_poi_distances().sort(dim=1).values
        return sorted_distances[:, self.agents_per_poi - 1]

    def _random_spot(self, n: int) -> Tensor:
        # n random positions inside the arena. Shape [n, 2]
        return (torch.rand(n, 2, device=self.world.device) * 2 - 1) * self.arena_size

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.pois,
            self.world,
            env_index,
            min_dist_between_entities=self.agent_radius * 2 + 0.05,
            x_bounds=(-self.arena_size, self.arena_size),
            y_bounds=(-self.arena_size, self.arena_size),
        )

        if env_index is None:
            self.previous_gap = self._gap_to_capture()
            self.catches[:] = 0
        else:
            self.previous_gap[env_index] = self._gap_to_capture()[env_index]
            self.catches[env_index] = 0

        # in reset_world_at:
        if env_index is None:
            self.goal_vel[:] = self._random_velocity(self.world.batch_dim)
        else:
            self.goal_vel[env_index] = self._random_velocity(1)[0]

    def _random_velocity(self, n_envs: int) -> Tensor:
        angle = torch.rand(n_envs, self.n_pois, device=self.world.device) * 2 * torch.pi
        return self.poi_speed * torch.stack([angle.cos(), angle.sin()], dim=-1)  # [n, POIs, 2]
        
    def pre_step(self):
        for i, poi in enumerate(self.pois):
            pos = poi.state.pos              # [batch, 2]
            v = self.goal_vel[:, i]          # [batch, 2]
    
            flip = ((pos >= self.arena_size) & (v > 0)) | ((pos <= -self.arena_size) & (v < 0))
            v = torch.where(flip, -v, v)
            self.goal_vel[:, i] = v
    
            poi.set_pos(pos + v * self.world.dt, batch_index=None)


    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def reward(self, agent: Agent):
        if agent is self.world.agents[0]:  # team reward: compute once per step
            distances = self._robot_poi_distances()  # [envs, robots, POIs]

            # Catches
            robots_on_poi = (distances < self.capture_range).sum(dim=1)  # [envs, POIs]
            caught = robots_on_poi >= self.agents_per_poi
            self.catches = caught.sum(dim=1).float()

            # Getting closer: how much each gap shrank since last step
            gap = self._gap_to_capture()
            self.closer_reward = (self.previous_gap - gap).sum(dim=1) * self.closer_reward_weight

            # Respawn caught POIs, then recompute gaps so the jump isn't penalised
            for j, poi in enumerate(self.pois):
                new_spot = self._random_spot(self.world.batch_dim)
                was_caught = caught[:, j].unsqueeze(-1)
                poi.set_pos(torch.where(was_caught, new_spot, poi.state.pos), batch_index=None)
            self.previous_gap = self._gap_to_capture()

            self.team_reward = self.catches * self.capture_reward + self.closer_reward + self.time_penalty

            # Bumping penalty (only for the robots involved)
            for robot in self.world.agents:
                robot.collision_reward[:] = 0
            # for i, a in enumerate(self.world.agents):
                # for b in self.world.agents[i + 1 :]:
            for i, a in enumerate(self.catchers):
                for b in self.catchers[i + 1 :]:
                    bumped = self.world.get_distance(a, b) <= 0.005
                    a.collision_reward[bumped] += self.collision_penalty
                    b.collision_reward[bumped] += self.collision_penalty

        return self.team_reward + agent.collision_reward

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def observation(self, agent: Agent):
        me = agent.state.pos
        # parts = []
        parts = [torch.full_like(me[:, :1], 1.0 if agent.is_scout else 0.0)]  # role: 1 = scout, 0 = catcher

        # POIs
        for poi in self.pois:
            offset = poi.state.pos - me
            can_see = (offset.norm(dim=-1, keepdim=True) < agent.sensing_range).float()
            parts += [offset * can_see, can_see]

        # other agents
        for teammate in self.world.agents:
            if teammate is agent:
                continue
            offset = teammate.state.pos - me
            can_see = (offset.norm(dim=-1, keepdim=True) < agent.sensing_range).float()
            parts += [offset * can_see, can_see]

        return {
            "obs": torch.cat(parts, dim=-1),
            "pos": agent.state.pos,  # for GNNs if used
            "vel": agent.state.vel,  # for GNNs if used
        }

    # ------------------------------------------------------------------ #
    # Logging / rendering
    # ------------------------------------------------------------------ #
    def info(self, agent: Agent) -> dict:
        return {
            "catches": self.catches,
            "closer_reward": self.closer_reward,
            "collision_reward": agent.collision_reward,
        }

    def extra_render(self, env_index: int = 0) -> list:
        from vmas.simulator import rendering

        def draw_circle(center, radius, color):
            circle = rendering.make_circle(radius, filled=False)
            move_to = rendering.Transform()
            move_to.set_translation(*center)
            circle.add_attr(move_to)
            circle.set_color(*color)
            return circle

        drawings = []
        for robot in self.world.agents:
            drawings.append(draw_circle(robot.state.pos[env_index], robot.sensing_range, Color.GRAY.value))
        for poi in self.pois:
            drawings.append(draw_circle(poi.state.pos[env_index], self.capture_range, Color.RED.value))

        # if self.dim_c > 0:
        for i, a in enumerate(self.world.agents):
            for b in self.world.agents[i + 1 :]:
                distance = (a.state.pos[env_index] - b.state.pos[env_index]).norm()
                if distance <= self.comm_range:
                    line = rendering.Line(a.state.pos[env_index], b.state.pos[env_index], width=1)
                    line.set_color(*Color.BLACK.value)
                    drawings.append(line)
        return drawings


if __name__ == "__main__":
    env = make_env(
        scenario=Scenario(),
        num_envs=8,
        device="cpu",
        seed=0,
        continuous_actions=True,
        max_steps=200,
        # n_agents=4,
        n_catchers=4,
        n_scouts=1,
        n_pois=3,
        # dim_c=0,
    )
    obs = env.reset()
    print("Robots:", len(env.agents))
    print("Obs shape (robot 0):", tuple(obs[0]["obs"].shape))
    print("Action space (robot 0):", env.action_space[0])

    total = torch.zeros(env.num_envs)
    for _ in range(200):
        obs, rews, dones, info = env.step([env.get_random_action(a) for a in env.agents])
        total += info[0]["catches"]
    print("Random policy catches per env over 200 steps:", total.tolist())
