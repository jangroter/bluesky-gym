from bluesky_zoo import merge_v0
from pettingzoo.test import parallel_api_test, parallel_test

env = merge_v0.MergeEnv(render_mode='human')
parallel_api_test(env, num_cycles=1_000_000)
