from queue import Queue
import threading as thr
from collections import namedtuple

from src.runners.MultiStepActorRunner import MultiStepActorRunner
from src.environments.MicroTestEnvironment import MicroTestEnvironment
from src.config.ConfigBase import ConfigBase

from src.util.reward_func import victory_if_zero_enemy
from src.util.state_process_func import process_game_state_to_dgl

import sys


class RunnerConfig(ConfigBase):
    def __init__(self, agent, name='runner', runner_conf=None, env_conf=None):
        super(RunnerConfig, self).__init__(name=name, runner=runner_conf, env=env_conf)
        self.env = {
            "map_name": 'training_scenario_4',
            "reward_func": victory_if_zero_enemy,
            "state_proc_func": process_game_state_to_dgl,
            "frame_skip_rate": 2,
            "realtime": False,
        }

        if sys.version_info[1] >= 7:  # python version == 3.7
            sample_spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done", "ret"],
                                     defaults=tuple([list() for _ in range(6)]))
        else:  # python version < 3.7
            sample_spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done", "ret"])
            sample_spec.__new__.__defaults__ = tuple([list() for _ in range(6)])

        self.runner = {
            'num_runners': 1,
            'n_hist_steps': 2,
            'sample_spec': sample_spec,
            'gamma': 0.9
        }

        self.agent = agent


class RunnerManager:
    def __init__(self, conf):
        self.num_runners = conf.runner['num_runners']
        self.agent = conf.agent
        self.runners = []

        self.sample_queue = Queue()
        self.eval_queue = Queue()

        for _ in range(self.num_runners):
            env = MicroTestEnvironment(**conf.env)

            self.runners.append(MultiStepActorRunner(
                env, conf.agent, conf.runner['sample_spec'],
                conf.runner['n_hist_steps'], conf.runner['gamma']))

    def sample(self, total_n):
        self.reset()

        threads = []
        for (n, runner) in zip(self._calc_n(total_n), self.runners):
            th = thr.Thread(target=runner.run_n_episodes,
                            args=(n, self.sample_queue))
            threads.append(th)

        for th in threads:
            th.start()

        for th in threads:
            th.join()

    def transfer_sample(self):
        trajectories = []
        while not self.sample_queue.empty():
            traj = self.sample_queue.get()
            trajectories.append(traj)

        self.agent.buffer.push_trajectories(trajectories)

    def evaluate(self, total_n):
        self.reset()
        self.set_eval_mode()

        threads = []
        for (n, runner) in zip(self._calc_n(total_n), self.runners):
            th = thr.Thread(target=runner.eval_n_episodes,
                            args=(n, self.eval_queue))
            threads.append(th)

        for th in threads:
            th.start()

        for th in threads:
            th.join()

        eval_dicts = []
        while not self.eval_queue.empty():
            eval_dict = self.eval_queue.get()
            eval_dicts.append(eval_dict)

        self.set_train_mode()
        return eval_dicts

    def set_eval_mode(self):
        for runner in self.runners:
            runner.set_eval_mode()

    def set_train_mode(self):
        for runner in self.runners:
            runner.set_train_mode()

    def close(self):
        for runner in self.runners:
            runner.close()

    def reset(self):
        for runner in self.runners:
            runner.reset()

    def _calc_n(self, total_n):
        div, remain = divmod(total_n, self.num_runners)
        ns = [div] * (self.num_runners - remain) + [div + 1] * remain
        return ns
