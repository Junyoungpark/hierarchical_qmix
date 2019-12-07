from src.environments.MicroTestEnvironment import MicroTestEnvironment
from src.runners.RunnerManager import RunnerConfig, RunnerManager
from src.agent.QmixAgent import QmixAgent, QmixAgentConfig

if __name__ == '__main__':

    conf = QmixAgentConfig()
    agent = QmixAgent(conf)

    runner_conf = RunnerConfig(agent=agent)
    runner = RunnerManager(runner_conf)

    iters = 0
    while iters < 100:
        iters += 1
        runner.sample(2)
        runner.transfer_sample()
        print("test")