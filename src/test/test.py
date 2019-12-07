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
        runner.sample(1)
        runner.transfer_sample()

        agent.to('cuda')
        agent.fit(device='cuda')
        agent.to('cpu')
