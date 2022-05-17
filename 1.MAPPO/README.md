# MAPPO in MPE environment
This is a concise Pytorch implementation of MAPPO in MPE environment(the multi-agent particle-world environment).<br />
This code only works in the environments where all agents are homogenous, such as 'Spread' in MPE. Here, all agents have the same dimension of observation space and action space.<br />

## How to use my code?
You can dircetly run 'MAPPO_main.py' in your own IDE.<br />

## Trainning environments
We train our MAPPO in 'simple_spread' in MPE environment.<br />

## Some details
Because the MPE environment is is relatively simple, we do not use RNN in 'actor' and 'critic' which can result in the better performence according to our experimental results.<br />
However, we also provide the implementation of using RNN. You can set 'use_rnn'=True in the hyperparameters setting, if you want to use RNN..<br />

## Trainning result
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/1.MAPPO/training%20result.png)

## Reference
[1] Yu C, Velu A, Vinitsky E, et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games[J]. arXiv preprint arXiv:2103.01955, 2021.<br />
[2] https://github.com/marlbenchmark/on-policy.<br />
