import wandb
import numpy as np

from src.runners.RunnerManager import RunnerConfig, RunnerManager
from src.agent.QmixAgent import QmixAgent, QmixAgentConfig

if __name__ == '__main__':

    exp_name = "qmix_refac"
    conf = QmixAgentConfig()
    agent = QmixAgent(conf)

    runner_conf = RunnerConfig(agent=agent)
    runner = RunnerManager(runner_conf)

    wandb.init(project="qmix3", name=exp_name)
    wandb.watch(agent)
    wandb.config.update(conf())

    iters = 0
    while iters < 100:
        iters += 1
        runner.sample(1)
        runner.transfer_sample()

        agent.to('cuda')
        fit_return_dict = agent.fit(device='cuda')
        agent.to('cpu')

        running_wrs = [runner.env.winning_ratio for runner in runner.runners]
        running_wr = np.mean(running_wrs)

        print(iters)
        print(fit_return_dict)

        wandb.log(fit_return_dict, step=iters)
        wandb.log({'train_winning_ratio': running_wr, 'epsilon': agent.brain.eps}, step=iters)

