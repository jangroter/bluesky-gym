import csv
import numbers
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Bookkeeping keys added by SB3/gymnasium wrappers, not by the environment itself.
EXCLUDED_INFO_KEYS = ('episode', 'terminal_observation', 'TimeLimit.truncated')


class CSVLoggerCallback(BaseCallback):
    """
    Logs one CSV row per finished episode, with the episode's final info dict.

    Works with both single environments and vectorised environments
    (e.g. SubprocVecEnv): SB3 wraps every environment in a VecEnv, so this
    callback always iterates over all parallel environment slots and captures
    episodes from each of them.

    In addition to the CSV, rolling means of episode infos are recorded to
    SB3's logger, making them show up in the verbose stdout table during
    training (and in TensorBoard when ``tensorboard_log`` is set on the model).

    :param log_dir: directory to write the CSV file to
    :param file_name: name of the CSV file
    :param monitor_keys: which info keys to surface in the verbose table.
        ``None`` (default) surfaces all numeric info keys, a tuple of key
        names surfaces only those, and an empty tuple ``()`` disables
        surfacing entirely. The CSV always contains all info keys.
    :param verbose: verbosity level of the callback itself
    """
    def __init__(self, log_dir, file_name='training_log.csv', monitor_keys=None, verbose=0):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, file_name)
        self.monitor_keys = monitor_keys
        self.initialized = False
        self.episode_count = 0
        # Rolling window of finished-episode infos, mirrors SB3's ep_info_buffer.
        self.ep_info_window = deque(maxlen=100)

    def _initialize(self, info_dict):
        """Set up CSV headers from the first finished episode's info dict."""
        self.info_keys = [key for key in info_dict.keys() if key not in EXCLUDED_INFO_KEYS]
        headers = ['timesteps', 'episodes', 'env_id'] + self.info_keys
        with open(self.log_file, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)
        self.initialized = True

    def _on_step(self) -> bool:
        for env_id, done in enumerate(self.locals['dones']):
            if not done:
                continue
            info_dict = self.locals['infos'][env_id]
            if not self.initialized:
                self._initialize(info_dict)
            self.episode_count += 1
            row = [self.num_timesteps, self.episode_count, env_id] + \
                  [info_dict.get(key, None) for key in self.info_keys]
            with open(self.log_file, mode='a', newline='') as f:
                csv.writer(f).writerow(row)

            self.ep_info_window.append(info_dict)
            self._record_rolling_means()

        return True

    def _record_rolling_means(self):
        """Record rolling means of episode infos to SB3's logger (verbose table / TensorBoard)."""
        for key in self.info_keys:
            if self.monitor_keys is not None and key not in self.monitor_keys:
                continue
            values = [ep_info[key] for ep_info in self.ep_info_window
                      if isinstance(ep_info.get(key), numbers.Number)]
            if values:
                self.logger.record(f'rollout/{key}_mean', np.mean(values))
