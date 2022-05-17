import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_len = args.episode_len
        self.batch_size = args.batch_size
        self.count = 0
        # create a buffer (dictionary)
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_len + 1, self.N, self.obs_dim]),
                       's': np.empty([self.batch_size, self.episode_len + 1, self.state_dim]),
                       'v_n': np.empty([self.batch_size, self.episode_len + 1, self.N]),
                       'a_n': np.empty([self.batch_size, self.episode_len, self.N]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_len, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_len, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_len, self.N])
                       }

    def store_episode(self, episode_obs_n, episode_s, episode_v_n, episode_a_n, episode_a_logprob_n, episode_r_n, episode_done_n):
        self.buffer['obs_n'][self.count] = episode_obs_n
        self.buffer['s'][self.count] = episode_s
        self.buffer['v_n'][self.count] = episode_v_n
        self.buffer['a_n'][self.count] = episode_a_n
        self.buffer['a_logprob_n'][self.count] = episode_a_logprob_n
        self.buffer['r_n'][self.count] = episode_r_n
        self.buffer['done_n'][self.count] = episode_done_n
        self.count += 1

    def numpy_to_tensor(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
