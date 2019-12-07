import numpy as np

from collections import deque, namedtuple
from src.config.ConfigBase import ConfigBase
from src.memory.Trajectory import Trajectory


class NstepMemoryConfig(ConfigBase):
    def __init__(self, name='nstepmemory', memory_conf=None):
        super(NstepMemoryConfig, self).__init__(name=name, memory=memory_conf)
        spec = namedtuple('exp_args', ['state', 'action', 'reward', 'next_state', 'done', 'ret'],
                          defaults=tuple([list() for _ in range(6)])),
        self.memory = {
            'spec': spec,
            'max_n_episodes': 3000,
            'gamma': 0.9,
            'max_traj_len': 30,
            'use_return': True,
            'N': 2
        }


class NstepMemory:
    def __init__(self, conf):
        self.conf = conf

        self.spec = conf.memory['spec']
        self.max_n_episodes = conf.memory['max_n_episodes']
        self.gamma = conf.memory['gamma']
        self.max_traj_len = conf.memory['max_traj_len']
        self.use_return = conf.memory['use_return']
        self.N = conf.memory['N']

        self.trajectories = deque(maxlen=self.max_n_episodes)
        self._cur_traj = Trajectory(gamma=self.gamma, max_len=self.max_traj_len, spec=self.spec)

    def push(self, sample):
        done = sample.done
        self._cur_traj.push(sample)
        if done:
            self.trajectories.append(self._cur_traj)
            self._cur_traj = Trajectory(gamma=self.gamma, max_len=self.max_traj_len, spec=self.spec)

    def push_trajectories(self, trajectories):
        for traj in trajectories:
            self.trajectories.append(traj)

    def sample(self, sample_size):
        len_trajectories = self.len_trajectories()
        num_samples_par_trajs = np.clip(len_trajectories - self.N, a_min=0, a_max=np.inf)
        num_samples_par_trajs = num_samples_par_trajs.astype(int)
        effective_num_samples = np.cumsum(num_samples_par_trajs)[-1]

        if sample_size > effective_num_samples:
            sample_size = effective_num_samples

        p = num_samples_par_trajs / np.sum(num_samples_par_trajs)
        samples_per_traj = np.random.multinomial(sample_size, p)

        hists = []
        states = []
        actions = []
        rewards = []
        next_hists = []
        next_states = []
        dones = []

        for traj_i, num_samples in enumerate(samples_per_traj):
            cur_traj = self.trajectories[traj_i]
            sample_is = np.random.choice(np.arange(self.N, cur_traj.length), num_samples)
            for sample_i in sample_is:
                hs, s, a, r, nhs, ns, d = self.sample_from_trajectory(trajectory_i=traj_i,
                                                                      sampling_index=sample_i)

                hists.append(hs)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                dones.append(d)
                next_hists.append(nhs)
                next_states.append(ns)

        return hists, states, actions, rewards, next_hists, next_states, dones

    def sample_from_trajectory(self, trajectory_i, sampling_index):

        traj = self.trajectories[trajectory_i]
        i = sampling_index

        hist = []
        for j in range(i - self.N, i):
            state = traj[j].state
            hist.append(state)

        next_hist = []
        for j in range(i - self.N + 1, i + 1):
            state = traj[j].state
            next_hist.append(state)

        state, action, reward, next_state, done, ret = traj[i]

        if self.use_return:
            reward = ret

        return hist, state, action, reward, next_hist, next_state, done

    def len_trajectories(self):
        return np.array([traj.length for traj in self.trajectories])
