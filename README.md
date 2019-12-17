# Hierarchical Qmix

a proof of concept project of Hierarchical Q-mix on Starcraft2 micro control environments.

The major contribution of this project is as follows:
1. Suggest a MARL model that can control arbitrary number of agents without additional training.
2. Suggest a Hierarchical Q-mixing strategy that is for learn to sub groups based on the input features.

## Requirements
* pytorch
* numpy
* DGL (Deep graph library)
* python-sc2 (supports <= 0.11.1 __python-sc2 is now DEPRECIATED__)
* wandb (a web based interactive logger)

## Simple Run with test code
You can run the test code with the following code
```
>>> python test/test.py
```

## Project structure
```
├───maps # contain SC2 maps -> Later copy this folder to your sc2 installation path
├───src
│   ├───agent 
│   │   └─── QmixAgent.py
│   │
│   ├───brain
│   │   │    Brainbase.py
│   │   └─── QmixBrain.py
│   │
│   ├───config
│   │   │    ConfigBase.py # base config class for mananginh hyperparameters
│   │   │    graph_config.py # hyper parameters used for constructing graph
│   │   └─── nn_config.py 
│   │
│   ├───environments # RL friendly wrapper of python-sc2 game runner
│   │   │    EnvironmentBase.py 
│   │   │    MicroTestEnvironment.py
│   │   └─── SC2BotAI.py
│   │
│   ├───memory # used for RL
│   │   │    MemoryBase.py
│   │   └─── Trajectory.py
│   │
│   ├───nn # neural networks 
│   │   │    activations.py
│   │   │    FeedForward.py
│   │   │    GCN.py
│   │   │    GraphConvolution.py
│   │   │    Linear.py
│   │   │    MLP.py
│   │   │    MultiStepInputGraphNetwork.py
│   │   │    RelationalGraphLayer.py
│   │   │    RelationalGraphNetwork.py
│   │   └─── RNNEncoder.py
│   ├───optim
│   │   │    Lookahead.py
│   │   └─── RAdam.py
│   │   
│   ├───rl # RL related modules + python-SC2 interface
│   │   │    ActionModules.py
│   │   │    MultiStepQnet.py
│   │   │    Qmixer.py
│   │   │    QmixNetwork.py
│   │   └─── Qnet.py
│   │
│   ├───runners # parallel execution helpers
│   │   │    MultiStepActorRunner.py 
│   │   │    RunnerBase.py
│   │   └─── RunnerManager.py 
│   │
│   ├───util
│   │   │    graph_util.py
│   │   │    HistoryManager.py
│   │   │    reward_func.py
│   │   │    sc2_config.py
│   │   │    sc2_util.py
│   │   │    state_process_func.py
│   │   └─── train_util.py
│
└───test
    │    context.py
    └─── test.py
```


