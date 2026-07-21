# Getting Started

A practical walk-through of the competition code: install, run the demo
scripts, understand the example training script, and customize the MDP. For the
rules and scoring see **[COMPETITION.md](COMPETITION.md)**.

---

## 1. Setup

Requires Python ≥ 3.10. The project is managed with [uv](https://docs.astral.sh/uv/)
(there is a `uv.lock`), but any virtualenv works.

```bash
# with uv (creates .venv from the lockfile)
uv sync

# then either prefix commands with `uv run`, or call the venv python directly:
.venv/bin/python test.py
```

The rest of this guide writes `python`; use `.venv/bin/python` (or `uv run
python`) to hit the project environment. Key dependencies: `bluesky-simulator`,
`gymnasium`, `pettingzoo`, `stable-baselines3`, `supersuit`, `torch`, `pygame`.

Two ways to run code, depending on where the file lives:

- **repo-root scripts** (`test.py`, `test_zoo.py`, `train_zoo.py`): `python test.py`
- **`scripts/` modules** (`evaluate_competition`, `record_competition_gifs`):
  run as a module from the repo root, e.g. `python -m scripts.evaluate_competition`.

---

## 2. See it run (the demo scripts)

### Single-agent — [`test.py`](../../test.py)

A quick visual check of `CompetitionEnv-v0`. It opens a pygame window, flies a
neutral (`[0, 0]` = keep heading and speed) policy for a few episodes, and prints
the objective scoring `info` dict at the end of each.

```bash
python test.py
```

### Multi-agent — [`test_zoo.py`](../../test_zoo.py)

The same idea for `CompetitionZooEnv` with `n_agents=10` (the competition
config): all agents fly straight, and each agent's metrics print as it leaves
the episode.

```bash
python test_zoo.py
```

> Running headless (no display)? Prefix with `SDL_VIDEODRIVER=dummy` to skip the
> window, or use the `render_mode=None` / `"rgb_array"` modes in your own code.

---

## 3. The example training script — [`train_zoo.py`](../../train_zoo.py)

`train_zoo.py` is a **simple but functional** learning setup for the
multi-agent environment. It is a *learnability sanity check* (mean per-agent
return before vs. after training), not a tuned solution — a good starting point
to build on.

```bash
python train_zoo.py 250000            # train, save to ppo_competition.zip
python train_zoo.py replay            # load it and render a greedy rollout
```

**Algorithm.** Stable-Baselines3 `PPO` with an `MlpPolicy`, using **parameter
sharing** — one shared policy drives all 10 (homogeneous) agents.

**The wrapper stack** (in `make_vec_env`) turns the PettingZoo `ParallelEnv`
into something SB3 can train on:

1. `FlattenObs` — flattens each agent's `Dict` observation into a 1-D `Box`
   (SuperSuit's `flatten_v0` only handles `Box`, not `Dict`), so a plain
   `MlpPolicy` works.
2. `black_death_v3` — keeps terminated agents (zero-padded) in the batch until
   the episode ends.
3. `pettingzoo_env_to_vec_env_v1` — exposes the N agents as an N-way vectorized
   env backed by a **single** underlying `ParallelEnv`.
4. `SB3VecEnvWrapper` + `VecMonitor` — adapt to the SB3 VecEnv API and record
   per-episode returns.

**Why a single underlying env matters:** BlueSky is a **process-global
singleton** — only one BlueSky-backed environment can be active per process.
That is why the script uses `pettingzoo_env_to_vec_env_v1` (one env, N agent
views) and **not** `concat_vec_envs_v1` (which would spin up multiple envs).

**Hyperparameters** (deliberately modest): `n_steps=512`, `batch_size=128`,
`gamma=0.99`, `gae_lambda=0.95`, `learning_rate=3e-4`, `ent_coef=0.0`,
`net_arch=[64, 64]`. Tune away.

> `N_AGENTS` is set to the competition value of 10; lower it for faster local
> iteration, but report/evaluate at 10.

---

## 4. Customize the MDP

You design your MDP by overriding three hooks — `_get_obs`, `_get_reward`,
`_get_action` — or by wrapping the env. Minimal reward-shaping example:

```python
import gymnasium as gym
import bluesky_gym
from bluesky_gym.envs.competition_env import CompetitionEnv

class MyEnv(CompetitionEnv):
    def _get_reward(self, ac_id):
        # your reward shaping (does NOT affect the scored metrics)
        base = super()._get_reward(ac_id)
        return base
```

Or via a standard wrapper (no subclass needed):

```python
bluesky_gym.register_envs()
env = gym.wrappers.TransformReward(gym.make("CompetitionEnv-v0"), lambda r: r)
```

If you change the **observation**, also rebuild the observation space in your
subclass `__init__` (or use an observation wrapper that declares the new space).
Both environment source files have detailed docstrings describing the contract:
[`competition_env.py`](../../bluesky_gym/envs/competition_env.py) and
[`competition.py`](../../bluesky_zoo/competition/competition.py).

### Train on the single-agent (required) track

There is no bundled single-agent trainer, but SB3 handles it directly. The
observation is a `Dict`, so either use `MultiInputPolicy`, or flatten it and use
`MlpPolicy`:

```python
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import PPO

bluesky_gym.register_envs()
env = gym.wrappers.FlattenObservation(gym.make("CompetitionEnv-v0"))  # Dict -> Box(131,)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_competition_sa")
```

(The env truncates itself at 3000 simulated seconds, so no `TimeLimit` wrapper
is needed.)

---

## 5. Reference

### Observation (shipped default)

A `Dict` of these components (flattened: **131** dims single-agent, **124**
multi-agent with `n_agents=10`). All bearings are cos/sin pairs; distances are
normalized.

| Component | Fields | Size |
| --- | --- | --- |
| Waypoint | `cos_drift`, `sin_drift`, `waypoint_distance` | 3 |
| Own airspeed | `airspeed` | 1 |
| Intruders (per neighbour) | `x_r`, `y_r`, `vx_r`, `vy_r`, `cos_track`, `sin_track`, `intruder_distance` | 7 × N |
| Obstacles (×5) | `obstacle_distance`, `obstacle_cos_bearing`, `obstacle_sin_bearing`, `obstacle_radius` | 4 × 5 |
| Sector (×12 + flag) | `sector_point_distance`, `sector_point_cos_bearing`, `sector_point_sin_bearing`, `inside_sector` | 37 |

N = number of intruders shown (single-agent 10; multi-agent `n_agents − 1` = 9).

### Action

`Box(-1, 1, shape=(2,))`: `action[0]` scales a heading change (× `d_heading`,
default 45°), `action[1]` scales a speed change (× `d_speed`, default ≈6.67 kt),
both relative to the current state. `[0, 0]` keeps heading and speed.

### Metrics

See the [metrics table in COMPETITION.md](COMPETITION.md#metrics-the-objective-score).

---

## 6. Evaluate & record

```bash
# official metrics over the first 1000 episodes of seed 42
python -m scripts.evaluate_competition --env sa                  # required
python -m scripts.evaluate_competition --env ma                  # stretch

# quick check + your trained policy (see load_policy in the script)
python -m scripts.evaluate_competition --env sa --episodes 20

# record GIFs (docs images, or your 5-scenario submission reel)
python -m scripts.record_competition_gifs                        # both doc GIFs
python -m scripts.record_competition_gifs --env sa --seeds 1 2 3 4 5 --out reel.gif
```

Both scripts have two clearly-marked edit points (an env factory and a
model/policy loader) — plug in your subclass and trained model there.
