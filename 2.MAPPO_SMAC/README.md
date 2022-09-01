# MAPPO in StarCraft II Environment
This is a concise Pytorch implementation of MAPPO in StarCraft II environment(SMAC-StarCraft Multi-Agent Challenge).<br />


## How to use my code?
You can dircetly run 'MAPPO_SMAC_main.py' in your own IDE.<br />

## Trainning environments
You can set the 'env_index' in the codes to change the maps in StarCraft II. Here, we train our code in 3 maps.<br />
env_index=0 represent '3m'<br />
env_index=1 represent '8m'<br />
env_index=2 represent '2s_3z'<br />

## Requirements
python==3.7.9<br />
numpy==1.19.4<br />
pytorch==1.12.0<br />
tensorboard==0.6.0<br />
[SMAC-StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)


## Trainning results
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/2.MAPPO_SMAC/MAPPO_SMAC_training_result.png)

## Reference
[1] Yu C, Velu A, Vinitsky E, et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games[J]. arXiv preprint arXiv:2103.01955, 2021.<br />
[2] [Official implementation of MAPPO](https://github.com/marlbenchmark/on-policy)<br />
[3] [EPyMARL](https://github.com/uoe-agents/epymarl)
