import wandb
import numpy as np

from src.runners.RunnerManager import RunnerConfig, RunnerManager
from src.agent.QmixAgent import QmixAgent, QmixAgentConfig

if __name__ == '__main__':

    exp_name = "Qmix_GCNSUP"

    conf = QmixAgentConfig()
    use_noisy_q = conf.brain.brain['use_noisy_q']

    agent = QmixAgent(conf)
    if use_noisy_q:
        agent.sample_noise()

    runner_conf = RunnerConfig(agent=agent)
    runner_conf.runner['num_runners'] = 1
    runner_conf.runner['n_hist_steps'] = conf.fit['hist_num_time_steps']
    runner = RunnerManager(runner_conf)

    wandb.init(project="qmix3", name=exp_name)
    wandb.watch(agent)
    wandb.config.update(conf())

    iters = 0
    while iters < 100:
        iters += 1
        runner.sample(6)
        runner.transfer_sample()

        agent.to('cuda')
        fit_return_dict = agent.fit(device='cuda')
        agent.to('cpu')

        running_wrs = [runner.env.winning_ratio for runner in runner.runners]
        running_wr = np.mean(running_wrs)

        fit_return_dict.update({'train_winning_ratio': running_wr, 'epsilon': agent.brain.eps})

        wandb.log(fit_return_dict, step=iters)

        if use_noisy_q:
            agent.sample_noise()
