"""
Scouts & Catchers: a VMAS scenario built for GNN policies.

THE TASK
    POIs (red dots) are scattered around the playground. A POI is "caught" when
    `catchers_needed` catchers are on it at the same time. It then respawns somewhere new.
    With `moving_pois=True`, every POI drifts in a straight line at `poi_speed` and
    bounces off the walls, so the team has to chase it.

THE TEAM
    Catchers (blue)  - the only robots that can catch POIs, but they're nearly blind
                       (small `catcher_sensing_range`).
    Scouts   (green) - see far (`scout_sensing_range`) but can't catch anything.
                       They fly overhead, so they never bump into anyone.

    So the team only does well if what scouts see reaches the catchers.

HOW INFORMATION MOVES: THE GNN
    This file has no message channel. Instead, the policy is a graph neural network:
      - every robot is a node in a graph,
      - robots closer than `comm_range` are connected by an edge,
      - each step, robots pass learned vectors along those edges before choosing an action.
    The graph itself is built by BenchMARL inside the policy (see train_gnn.py),
    using the "pos" key we return from observation().

WHAT EACH ROBOT OBSERVES (its "node features")
    obs = [ role,  POI slot 1,  POI slot 2, ... ]
            role     : 1.0 for a scout, 0.0 for a catcher. All robots share ONE network,
                       and this flag lets it act differently for each role.
            POI slot : (dx, dy, visible) for one of the nearest POIs this robot can see.
                       There are `n_poi_slots` slots, nearest first. Empty slots are zeros.
    plus two separate keys the GNN uses:
            pos      : my (x, y). Used to decide who's connected, and as a feature.
            vel      : my velocity. Used as a feature and for relative-velocity edge features.

    Note what's NOT in obs: anything about teammates. The size of obs is the same
    no matter how many robots there are. Teammate information only arrives through the graph.

Run this file to check that it works:  python gnn_scenario.py
"""

import torch
from torch import Tensor

from vmas import make_env
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        batch_dim = how many copies of the world run in parallel (VMAS simulates them all at once).
        """
        # settings

        # team info 
        self.n_catchers = kwargs.pop("n_catchers", 4)
        self.n_scouts = kwargs.pop("n_scouts", 2)
        self.catcher_sensing_range = kwargs.pop("catcher_sensing_range", 0.15)
        self.scout_sensing_range = kwargs.pop("scout_sensing_range", 0.8)
        self.comm_range = kwargs.pop("comm_range", 1.0)
        # Speed multipliers: 1.0 = normal (top speed ~0.4/s), 0.5 = half as fast
        self.catcher_speed = kwargs.pop("catcher_speed", 1.0)
        self.scout_speed = kwargs.pop("scout_speed", 1.0)

        # POIs
        self.n_pois = kwargs.pop("n_pois", 3)
        self.n_poi_slots = kwargs.pop("n_poi_slots", 3)  # how many POIs one robot can report in its obs
        self.catchers_needed = kwargs.pop("catchers_needed", 2)
        self.capture_range = kwargs.pop("capture_range", 0.15)
        # Moving POIs: each POI drifts in a random direction at poi_speed and bounces off the walls
        self.moving_pois = kwargs.pop("moving_pois", False)
        self.poi_speed = kwargs.pop("poi_speed", 0.05)  # distance per second (a catcher at speed 1.0 does ~0.4)

        # rewards
        self.capture_reward = kwargs.pop("capture_reward", 1.0)
        self.closer_reward_weight = kwargs.pop("closer_reward_weight", 1.0)
        # What the "getting closer" bonus measures:
        #   "gap"  - for each POI, the catcher that would complete the catch (the 2nd closest when
        #            catchers_needed=2). Extra catchers earn nothing for coming closer.
        #   "each" - every catcher, toward its own nearest POI. Nobody gets a free ride.
        self.closer_reward_mode = kwargs.pop("closer_reward_mode", "gap")
        self.time_penalty = kwargs.pop("time_penalty", -0.01)
        self.collision_penalty = kwargs.pop("collision_penalty", -0.1)

        # playground size
        self.playground_size = kwargs.pop("playground_size", 1.0)
        self.robot_radius = 0.05

        ScenarioUtils.check_kwargs_consumed(kwargs)  # errors on typos in setting names

        world = World(batch_dim, device, x_semidim=self.playground_size, y_semidim=self.playground_size)

        # robots
        # catchers come first (agent_0 .. agent_3), then scouts (agent_4, agent_5).
        self.catchers, self.scouts = [], []
        for i in range(self.n_catchers + self.n_scouts):
            is_scout = i >= self.n_catchers
            robot = Agent(
                name=f"agent_{i}",
                shape=Sphere(radius=self.robot_radius),
                color=Color.GREEN if is_scout else Color.BLUE,
                collide=not is_scout,  # scouts is free to move over everything
                # u_multiplier scales the push: force = action * u_multiplier.
                # The action is still in [-1, 1], but the same action pushes a slower robot less,
                # so its top speed drops by the same factor.
                u_multiplier=self.scout_speed if is_scout else self.catcher_speed,
            )
            
            robot.is_scout = is_scout
            robot.sensing_range = self.scout_sensing_range if is_scout else self.catcher_sensing_range
            robot.collision_reward = torch.zeros(batch_dim, device=device)

            world.add_agent(robot)
            (self.scouts if is_scout else self.catchers).append(robot)

        # POIs
        self.pois = []
        for i in range(self.n_pois):
            poi = Landmark(name=f"poi_{i}", shape=Sphere(radius=0.03), color=Color.RED, collide=False)
            world.add_landmark(poi)
            self.pois.append(poi)
        self.poi_vel = torch.zeros(batch_dim, self.n_pois, 2, device=device)  # [copies, POIs, (vx, vy)]

        self.catches = torch.zeros(batch_dim, device=device)          # POIs caught this step
        self.closer_reward = torch.zeros(batch_dim, device=device)    # "getting closer" bonus this step
        self.team_reward = torch.zeros(batch_dim, device=device)      # total shared reward this step
        self.previous_progress = None  # distances the "getting closer" bonus tracks; set in reset_world_at()

        return world


    def reset_world_at(self, env_index: int = None):
        """env_index=None resets every copy of the world, otherwise only copy number env_index."""

        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.pois,
            self.world,
            env_index,
            min_dist_between_entities=self.robot_radius * 2 + 0.05,
            x_bounds=(-self.playground_size, self.playground_size),
            y_bounds=(-self.playground_size, self.playground_size),
        )

        # Remember the starting distances, so the first "getting closer" reward is measured from here
        if env_index is None:
            self.previous_progress = self._progress_distances()
            self.catches[:] = 0
        else:
            self.previous_progress[env_index] = self._progress_distances()[env_index]
            self.catches[env_index] = 0

        # Give every POI a new random direction (only used when moving_pois is on)
        if env_index is None:
            self.poi_vel[:] = self._random_poi_velocities(self.world.batch_dim)
        else:
            self.poi_vel[env_index] = self._random_poi_velocities(1)[0]


    def pre_step(self):
        """VMAS calls this every step, before the physics. We use it to move the POIs."""
        if not self.moving_pois:
            return
        for j, poi in enumerate(self.pois):
            pos = poi.state.pos        # [copies, 2]
            vel = self.poi_vel[:, j]   # [copies, 2]
            # Bounce: past a wall and still heading out -> flip that direction (x and y separately)
            past_wall = ((pos >= self.playground_size) & (vel > 0)) | ((pos <= -self.playground_size) & (vel < 0))
            vel = torch.where(past_wall, -vel, vel)
            self.poi_vel[:, j] = vel
            poi.set_pos(pos + vel * self.world.dt, batch_index=None)  # dt = seconds per step (0.1)


    def reward(self, agent: Agent) -> Tensor:
        """
        Team reward = catches + getting closer + time penalty   (same for every robot)
        Own reward  = bump penalty                              (only for the catchers involved)
        """

        if agent is self.world.agents[0]:
            distances = self._catcher_poi_distances()  # [copies, catchers, POIs]

            # catches
            catchers_on_poi = (distances < self.capture_range).sum(dim=1)  # [copies, POIs]
            caught = catchers_on_poi >= self.catchers_needed                # [copies, POIs] True/False
            self.catches = caught.sum(dim=1).float()                        # [copies]

            # tracked distances shrank since last step -> positive reward, grew -> negative.
            progress = self._progress_distances()
            self.closer_reward = (self.previous_progress - progress).sum(dim=1) * self.closer_reward_weight

            # caught POIs jump to a random new spot (only in the copies where they were caught)
            for j, poi in enumerate(self.pois):
                new_spot = self._random_positions(self.world.batch_dim)
                poi.set_pos(torch.where(caught[:, j : j + 1], new_spot, poi.state.pos), batch_index=None)
            # Measure again AFTER respawning, so the jump doesn't count as "falling behind"
            self.previous_progress = self._progress_distances()

            self.team_reward = self.catches * self.capture_reward + self.closer_reward + self.time_penalty

            # collision penalty
            for robot in self.world.agents:
                robot.collision_reward[:] = 0
            for i, a in enumerate(self.catchers):
                for b in self.catchers[i + 1 :]:
                    bumped = self.world.get_distance(a, b) <= 0.005
                    a.collision_reward[bumped] += self.collision_penalty
                    b.collision_reward[bumped] += self.collision_penalty

        return self.team_reward + agent.collision_reward


    def observation(self, agent: Agent) -> dict:
        me = agent.state.pos  # [copies, 2]

        # distances to all POIs
        poi_positions = torch.stack([poi.state.pos for poi in self.pois], dim=1)  # [copies, POIs, 2]
        offsets = poi_positions - me.unsqueeze(1)                                # [copies, POIs, 2]
        distances = offsets.norm(dim=-1)                                         # [copies, POIs]

        # keep only the ones we can see
        distances = distances.masked_fill(distances >= agent.sensing_range, float("inf"))
        sorted_dist, order = distances.sort(dim=1)

        slots = []
        for s in range(self.n_poi_slots):
            if s < self.n_pois:
                visible = torch.isfinite(sorted_dist[:, s : s + 1]).float()  # [copies, 1]: 1 = I see it
                idx = order[:, s].view(-1, 1, 1).expand(-1, 1, 2)             # which POI goes in this slot
                offset = offsets.gather(1, idx).squeeze(1) * visible          # [copies, 2], zeros if unseen
            else:  # more slots than POIs: always empty
                visible = torch.zeros_like(me[:, :1])
                offset = torch.zeros_like(me)
            slots += [offset, visible]

        role = torch.full_like(me[:, :1], 1.0 if agent.is_scout else 0.0)  # [copies, 1]

        return {
            "obs": torch.cat([role] + slots, dim=-1),  # [copies, 1 + 3 * n_poi_slots]
            "pos": agent.state.pos,                     # for building the graph
            "vel": agent.state.vel,                     # for edge features
        }

    # ====================================================================== #
    # 5. EXTRAS: logging and drawing
    # ====================================================================== #
    def info(self, agent: Agent) -> dict:
        # BenchMARL logs these, and watch_gnn.py uses "catches" to count POIs caught
        return {"catches": self.catches, "closer_reward": self.closer_reward}

    def extra_render(self, env_index: int = 0) -> list:
        """Extra things to draw: sensing rings, capture rings, and the graph's edges."""
        from vmas.simulator import rendering

        def ring(center, radius, color):
            circle = rendering.make_circle(radius, filled=False)
            transform = rendering.Transform()
            transform.set_translation(*center)
            circle.add_attr(transform)
            circle.set_color(*color)
            return circle

        drawings = []
        for robot in self.world.agents:  # gray ring = what this robot can see
            drawings.append(ring(robot.state.pos[env_index], robot.sensing_range, Color.GRAY.value))
        for poi in self.pois:  # red ring = capture zone
            drawings.append(ring(poi.state.pos[env_index], self.capture_range, Color.RED.value))

        # Black line = these two robots are connected in the GNN's graph
        robots = self.world.agents
        for i, a in enumerate(robots):
            for b in robots[i + 1 :]:
                pa, pb = a.state.pos[env_index], b.state.pos[env_index]
                if (pa - pb).norm() <= self.comm_range:
                    line = rendering.Line(pa, pb, width=1)
                    line.set_color(*Color.BLACK.value)
                    drawings.append(line)
        return drawings

    # ====================================================================== #
    # Helpers
    # ====================================================================== #
    def _catcher_poi_distances(self) -> Tensor:
        """Distance from every catcher to every POI. Shape [copies, catchers, POIs]."""
        catcher_positions = torch.stack([c.state.pos for c in self.catchers], dim=1)
        poi_positions = torch.stack([p.state.pos for p in self.pois], dim=1)
        return torch.cdist(catcher_positions, poi_positions)

    def _gap_to_capture(self) -> Tensor:
        """
        For each POI: distance of the k-th closest catcher (k = catchers_needed).
        That's the catcher that would complete the catch. Shape [copies, POIs].
        """
        sorted_distances = self._catcher_poi_distances().sort(dim=1).values
        return sorted_distances[:, self.catchers_needed - 1]

    def _progress_distances(self) -> Tensor:
        """
        The distances the "getting closer" bonus tracks (the bonus = how much they shrank).
          "gap":  for each POI, the catcher that would complete the catch.  Shape [copies, POIs]
          "each": for each catcher, its nearest POI.                        Shape [copies, catchers]
        """
        if self.closer_reward_mode == "gap":
            return self._gap_to_capture()
        if self.closer_reward_mode == "each":
            return self._catcher_poi_distances().min(dim=2).values  # nearest POI per catcher
        raise ValueError(f"closer_reward_mode must be 'gap' or 'each', got {self.closer_reward_mode!r}")

    def _random_poi_velocities(self, n: int) -> Tensor:
        """A random direction at poi_speed for every POI in n copies. Shape [n, POIs, 2]."""
        angle = torch.rand(n, self.n_pois, device=self.world.device) * 2 * torch.pi
        return self.poi_speed * torch.stack([angle.cos(), angle.sin()], dim=-1)

    def _random_positions(self, n: int) -> Tensor:
        """n random (x, y) points inside the playground. Shape [n, 2]."""
        return (torch.rand(n, 2, device=self.world.device) * 2 - 1) * self.playground_size


# ========================================================================== #
# Quick check: python gnn_scenario.py
# ========================================================================== #
if __name__ == "__main__":
    env = make_env(Scenario(), num_envs=8, device="cpu", seed=0, continuous_actions=True, max_steps=200)
    obs = env.reset()

    print("Robots:", [a.name for a in env.agents], "(last", env.scenario.n_scouts, "are scouts)")
    print("obs per robot:", obs[0]["obs"].shape[-1], "numbers = role + 3 per POI slot")
    print("  catcher 0 sees:", [round(v, 2) for v in obs[0]["obs"][0].tolist()])
    print("  scout 0 sees:  ", [round(v, 2) for v in obs[env.scenario.n_catchers]["obs"][0].tolist()])
    print("Action per robot:", env.action_space[0], "-> push in x and y")

    total = torch.zeros(env.num_envs)
    for _ in range(200):
        obs, rewards, dones, info = env.step([env.get_random_action(a) for a in env.agents])
        total += info[0]["catches"]
    print("Random actions, POIs caught per copy in 200 steps:", total.tolist())
