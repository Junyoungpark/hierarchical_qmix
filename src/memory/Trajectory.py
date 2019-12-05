from collections import deque
import numpy as np


class Trajectory:
    def __init__(self, gamma, max_len, spec):
        self.spec = spec
        self.gamma = gamma
        self._trajectory = deque(maxlen=max_len)

    def push(self, sample):
        assert self.spec.fields == sample.fields
        self._trajectory.append(sample)
        done = sample.done
        if done:
            next_reward = sample.reward
            if sample.next_state.number_of_nodes() == 0:
                self._trajectory.pop()
                sample = self._trajectory.pop()
                state = sample.state
                action = sample.action
                reward = next_reward
                next_state = sample.next_state
                done = True
                ret = sample.ret
                self._trajectory.append(self.spec(state, action, reward, next_state, done, ret))
            self.compute_return()

    def compute_return(self):
        rewards = [sample.reward for sample in self._trajectory]
        returns = np.zeros_like(rewards, dtype=float)

        returns[-1] = rewards[-1]

        for i, reward in enumerate(reversed(rewards[:-1])):
            backward_index = self.length - 1 - i
            returns[backward_index - 1] = rewards[backward_index - 1] + self.gamma * returns[backward_index]

        # Set return values to the samples
        for i in range(self.length):
            sample = self._trajectory.popleft()

            state = sample.state
            action = sample.action
            reward = sample.reward
            next_state = sample.next_state
            done = sample.done
            ret = returns[i]
            self._trajectory.append(self.spec(state, action, reward, next_state, done, ret))

    @property
    def length(self):
        return len(self._trajectory)