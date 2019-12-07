class RunnerBase:

    def __init__(self, env, agent, sample_spec):
        self.env = env
        self.agent = agent
        self.sample_spec = sample_spec

    def step(self, env_action):
        return self.env.step(env_action)

    def observe(self):
        return self.env.observe()

    def run_1_episode(self):
        """
        :return: Trajectory object
        """
        raise NotImplementedError("This method will be implemented in the child class")

    def run_n_episodes(self, n, queue):
        for _ in range(n):
            trajectory = self.run_1_episode()
            queue.put(trajectory)

    def eval_1_episode(self):
        raise NotImplementedError("This method will be implemented in the child class")

    def eval_n_episodes(self, n, queue):
        for _ in range(n):
            eval_dict = self.eval_1_episode()
            queue.put(eval_dict)

    def set_eval_mode(self):
        self.agent.eval()

    def set_train_mode(self):
        self.agent.train()

    def reset(self):
        self.env.close()
        self.env.reset()

    def close(self):
        self.env.close()
