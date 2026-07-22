# BlueSky-Gym
A gymnasium style library for standardized Reinforcement Learning research in Air Traffic Management developed in Python.
Built on [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) and The Farama Foundation's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

<p align="center">
    <img src="https://github.com/user-attachments/assets/6ae83579-78af-4cb7-8096-3a10af54a5c5" width=50% height=50%><br/>
    <em>An example trained agent attempting the merge environment available in BlueSky-Gym.</em>
</p>

For a complete list of the currently available environments click [here](bluesky_gym/envs/README.md)

## 🏆 RL Competition

This repository hosts an **air traffic control RL competition** organized by the TU Delft and [AI4REAL-NET](https://ai4realnet.eu/): 
Guide aircraft to their goals while avoiding conflicts, restricted areas, and sector exits in
a single-agent and a multi-agent environment.

<p align="center">
    <img src="docs/media/competition/single_agent.gif" width="25%">
    <img src="docs/media/competition/multi_agent.gif" width="25%"><br/>
    <em>Left: single-agent competition env. Right: multi-agent competition env (neutral policy shown).</em>
</p>

- **[Competition outline & rules](docs/competition/COMPETITION.md)** — what you may
  change, what is fixed, the metrics, and how you are scored.
- **[Getting started guide](docs/competition/GETTING_STARTED.md)** — install, run
  the demo scripts, understand `train_zoo.py`, and customize the MDP.

## Submission & logistics

- **Deadline:** 30th of November 2026
- **How to submit:** Submit a technical report including metrics and video(s) of the trained policy to the email below, your latest submission will be graded
- **Team size:** No limit
- **Prizes:** Prize money depending on funding, will be updated once known
- **Eligibility:** Everyone
- **Questions / contact:** Join our [Discord](https://discord.gg/s7CdxcSX) or email: d.j.groot@tudelft.nl

## Installation

`pip install bluesky-gym`

Note that the pip package is `bluesky-gym`, for usage however, import as `bluesky_gym`.

## Usage
Using the environments follows the standard API from Gymnasium, an example of which is given below:

```python
import gymnasium as gym
import bluesky_gym
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
```

Additionally you can directly use algorithms from standardized libraries such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) or [RLlib](https://docs.ray.io/en/latest/rllib/index.html) to train a model:

```python
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode=None)
model = DDPG("MultiInputPolicy",env)
model.learn(total_timesteps=2e6)
model.save()
```

For more info, please refer to the [workshop slides](https://docs.google.com/presentation/d/1Jpwdrx__OMdgHWtQ1yCVQyxsdDFk2ieX/edit?usp=drive_link&ouid=109800667545002770848&rtpof=true&sd=true) that provide additional information on BlueSky-Gym and how to use it for your own needs.

## Contributing and Assistance
If you would like to contribute to BlueSky-Gym or need assistance in setting up or creating your own environments, do not hesitate to open an issue or reach out to one of us via the BlueSky-Gym [Discord](https://discord.gg/s7CdxcSX).
Additionally you can have a look at the [roadmap](https://github.com/TUDelft-CNS-ATM/bluesky-gym/issues/24) for inspiration on where you can contribute and to get an idea of the direction BlueSky-Gym is going.


## Citing

If you use BlueSky-Gym in your work, please cite it using:
```bibtex
@misc{bluesky-gym,
  author = {Groot, DJ and Leto, G and Vlaskin, A and Moec, A and Ellerbroek, J},
  title = {BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications},
  year = {2024},
  journal = {SESAR Innovation Days 2024},
}
```

List of publications & preprints using `BlueSky-Gym` (please open a pull request to add missing entries):
*   _missing entry_
