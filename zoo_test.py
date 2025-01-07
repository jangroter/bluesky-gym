from bluesky_zoo import sector_cr_v0
from pettingzoo.test import parallel_api_test, parallel_test

env = sector_cr_v0.SectorCR(render_mode='human')
parallel_api_test(env, num_cycles=1_000_000)
