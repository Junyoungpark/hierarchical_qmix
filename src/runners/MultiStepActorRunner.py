import numpy as np

from src.runners.RunnerBase import RunnerBase
from src.util.HistoryManager import HistoryManager
from src.memory.Trajectory import Trajectory

from src.config.graph_config import NODE_ALLY, NODE_ENEMY
from src.util.graph_util import get_filtered_node_index_by_type
from src.environments.MicroTestEnvironment import Status


class MultiStepActorRunner(RunnerBase):

    def __init__(self, env, agent, sample_spec, n_steps, gamma=1.0):
        super(MultiStepActorRunner, self).__init__(
            env=env, agent=agent, sample_spec=sample_spec)
        self.history_manager = HistoryManager(
            n_hist_steps=n_steps, init_graph=None)

        self.gamma = gamma

    def run_1_episode(self):
        trajectory = Trajectory(gamma=self.gamma, spec=self.sample_spec)
        # the first frame of each episode
        curr_state_dict = self.env.observe()
        curr_graph = curr_state_dict['g']
        self.history_manager.reset(curr_graph)

        while True:
            curr_state_dict = self.env.observe()
            curr_graph = curr_state_dict['g']

            tag2unit_dict = curr_state_dict['tag2unit_dict']
            hist_graph = self.history_manager.get_hist()

            nn_action, sc2_action, _ = self.agent.get_action(hist_graph=hist_graph, curr_graph=curr_graph,
                                                             tag2unit_dict=tag2unit_dict)

            next_state_dict, reward, done = self.env.step(sc2_action)
            next_graph = next_state_dict['g']
            experience = self.sample_spec(
                curr_graph, nn_action, reward, next_graph, done)

            trajectory.push(experience)
            self.history_manager.append(next_graph)
            if done:
                break

        return trajectory

    def eval_1_episode(self):
        # expected return
        # dictionary = {'name': (str), 'win': (bool), 'sum_reward': (float)}

        running_wr = self.env.winning_ratio

        env_name = self.env.name
        traj = self.run_1_episode()

        last_graph = traj[-1].state
        num_allies = get_filtered_node_index_by_type(last_graph, NODE_ALLY).size()
        num_enemies = get_filtered_node_index_by_type(last_graph, NODE_ENEMY).size()

        sum_reward = np.sum([exp.reward for exp in traj._trajectory])

        eval_dict = dict()
        eval_dict['name'] = env_name
        eval_dict['win'] = num_allies > num_enemies
        eval_dict['sum_reward'] = sum_reward

        self.env.winning_ratio = running_wr
        return eval_dict

    def set_train_mode(self):
        self.agent.train()

    def set_eval_mode(self):
        self.agent.eval()

    def reset(self):
        if self.env.status == Status.END:
            self.env.reset()

    def close(self):
        self.env.close()
