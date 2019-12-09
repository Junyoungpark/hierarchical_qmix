import context

import wandb
import numpy as np

from src.runners.RunnerManager import RunnerConfig, RunnerManager
from src.agent.QmixAgent import QmixAgent, QmixAgentConfig
from src.rl.QmixNetwork import QmixerConfig

if __name__ == '__main__':

    exp_name = "DEBUG"

    # test variable
    num_runners = 1
    num_samples = 10
    use_gcn = True
    rectifier = 'softplus'
    use_clipped_score = True

    conf = QmixAgentConfig()
    use_noisy_q = conf.brain.brain['use_noisy_q']
    gamma = conf.brain.brain['gamma']
    conf.mixer.qmix['rectifier'] = rectifier
    conf.mixer.qmix['use_gcn'] = use_gcn
    conf.mixer.submixer.mixer['use_clipped_score'] = use_clipped_score

    agent = QmixAgent(conf)
    if use_noisy_q:
        agent.sample_noise()

    runner_conf = RunnerConfig(agent=agent)
    runner_conf.runner['num_runners'] = num_runners
    runner_conf.runner['gamma'] = gamma
    runner_conf.runner['n_hist_steps'] = conf.fit['hist_num_time_steps']
    runner = RunnerManager(runner_conf)

    wandb.init(project="qmix3", name=exp_name)
    wandb.watch(agent)
    wandb.config.update(conf())

    iters = 0
    while iters < 1000:
        iters += 1
        runner.sample(num_samples)
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
