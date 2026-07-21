"""
Record GIFs of the competition environments.

Runs headless (no window) and captures rgb_array frames, so it works over SSH
or in CI. By default it regenerates the two GIFs embedded in the docs:

    docs/media/competition/single_agent.gif   (CompetitionEnv-v0)
    docs/media/competition/multi_agent.gif    (CompetitionZooEnv, n_agents=10)

with a neutral (fly-straight) policy — the initial heading points at the goal,
so this still shows the task, the other aircraft/intruders, the obstacles and
the sector.

It also doubles as the tool competitors can use to produce their required
short video of behaviour on a handful of scenarios: pass your own seeds (and,
if you like, plug your trained policy into ``policy_from_model`` below).

Run it as a module from the repo root (like the other scripts/ files):

    # regenerate both doc GIFs
    python -m scripts.record_competition_gifs

    # one scenario of the multi-agent env at a chosen seed
    python -m scripts.record_competition_gifs --env ma --seeds 7 --out my_ma.gif

    # a 5-scenario reel of the single-agent env (cherry-pick the seeds)
    python -m scripts.record_competition_gifs --env sa --seeds 1 2 3 4 5 --out reel.gif
"""

import argparse
import os

# Force pygame's headless "dummy" video backend so no window opens (works over
# SSH / in CI). setdefault => you can still override SDL_VIDEODRIVER yourself.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np                                  # noqa: E402
from PIL import Image                               # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SA_GIF = os.path.join(REPO_ROOT, "docs", "media", "competition", "single_agent.gif")
DEFAULT_MA_GIF = os.path.join(REPO_ROOT, "docs", "media", "competition", "multi_agent.gif")


def policy_from_model(model_path):
    """Return an ``act(obs) -> action`` callable, or None for a neutral policy.

    Competitors: load your trained policy here to record its behaviour, e.g.
    ``model = PPO.load(model_path); return lambda obs: model.predict(obs, deterministic=True)[0]``.
    The default (``model_path is None``) flies straight to the goal.
    """
    if model_path is None:
        return None
    raise NotImplementedError("plug your trained policy into policy_from_model()")


def rollout_single_agent(seed, act=None, max_steps=400):
    """Collect rgb_array frames for one CompetitionEnv-v0 episode."""
    import gymnasium as gym
    import bluesky_gym
    bluesky_gym.register_envs()

    env = gym.make("CompetitionEnv-v0", render_mode="rgb_array")
    frames = []
    obs, _ = env.reset(seed=seed)
    frames.append(env.render())
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        action = np.zeros(2, dtype=np.float32) if act is None else act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        steps += 1
    env.close()
    return frames


def rollout_multi_agent(seed, n_agents=10, act=None, max_steps=400):
    """Collect rgb_array frames for one CompetitionZooEnv episode."""
    from bluesky_zoo.competition_v0 import CompetitionZooEnv

    env = CompetitionZooEnv(render_mode="rgb_array", n_agents=n_agents)
    frames = []
    observations, _ = env.reset(seed=seed)
    frames.append(env.render())
    steps = 0
    while env.agents and steps < max_steps:
        if act is None:
            actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
        else:
            actions = {a: act(observations[a]) for a in env.agents}
        observations, _, _, _, _ = env.step(actions)
        frames.append(env.render())
        steps += 1
    env.close()
    return frames


def save_gif(frames, out_path, size=256, max_frames=120, duration=80):
    """Downscale, subsample and encode frames as a looping GIF."""
    if not frames:
        raise ValueError("no frames to save")
    if len(frames) > max_frames:                    # keep the file small
        stride = int(np.ceil(len(frames) / max_frames))
        frames = frames[::stride]
    images = [Image.fromarray(f).resize((size, size), Image.BILINEAR) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   loop=0, duration=duration, optimize=True)
    print(f"wrote {out_path}  ({len(images)} frames)")


def record(env_kind, seeds, out_path, n_agents=10, model_path=None, **gif_kwargs):
    """Record one or more scenarios (concatenated) into a single GIF."""
    act = policy_from_model(model_path)
    frames = []
    for seed in seeds:
        if env_kind == "sa":
            frames += rollout_single_agent(seed, act=act)
        else:
            frames += rollout_multi_agent(seed, n_agents=n_agents, act=act)
    save_gif(frames, out_path, **gif_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["sa", "ma", "both"], default="both",
                        help="single-agent, multi-agent, or both doc GIFs (default: both)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="scenario seed(s) to record (concatenated into one GIF)")
    parser.add_argument("--n-agents", type=int, default=10,
                        help="agents for the multi-agent env (competition config: 10)")
    parser.add_argument("--out", default=None, help="output GIF path (single --env only)")
    parser.add_argument("--model", default=None,
                        help="path to a trained model (see policy_from_model)")
    args = parser.parse_args()

    if args.env == "both":
        if args.out is not None:
            parser.error("--out requires a single --env (sa or ma)")
        record("sa", args.seeds or [3], DEFAULT_SA_GIF, model_path=args.model)
        record("ma", args.seeds or [1], DEFAULT_MA_GIF,
               n_agents=args.n_agents, model_path=args.model)
    else:
        default_out = DEFAULT_SA_GIF if args.env == "sa" else DEFAULT_MA_GIF
        record(args.env, args.seeds or [1], args.out or default_out,
               n_agents=args.n_agents, model_path=args.model)


if __name__ == "__main__":
    main()
