from gymnasium.envs.registration import register
from .utils import *
 
def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    register(
        id="DescentEnv-v0",
        entry_point="bluesky_gym.envs.descent_env:DescentEnv",
        max_episode_steps=300,
    )

    register(
        id="PlanWaypointEnv-v0",
        entry_point="bluesky_gym.envs.plan_waypoint_env:PlanWaypointEnv",
        max_episode_steps=300,
    )

    register(
        id="HorizontalCREnv-v0",
        entry_point="bluesky_gym.envs.horizontal_cr_env:HorizontalCREnv",
        max_episode_steps=300,
    )

    register(
        id="VerticalCREnv-v0",
        entry_point="bluesky_gym.envs.vertical_cr_env:VerticalCREnv",
        max_episode_steps=300,
    )

    register(
        id="SectorCREnv-v0",
        entry_point="bluesky_gym.envs.sector_cr_env:SectorCREnv",
        max_episode_steps=200,
    )

    register(
        id="StaticObstacleEnv-v0",
        entry_point="bluesky_gym.envs.static_obstacle_env:StaticObstacleEnv",
        max_episode_steps=100,
    )

    register(
        id="MergeEnv-v0",
        entry_point="bluesky_gym.envs.merge_env:MergeEnv",
        max_episode_steps=50,
    )

    # No max_episode_steps: CompetitionEnv truncates itself on simulated time
    # (episode_time_limit), so the episode budget is independent of action_frequency.
    register(
        id="CompetitionEnv-v0",
        entry_point="bluesky_gym.envs.competition_env:CompetitionEnv",
    )
    