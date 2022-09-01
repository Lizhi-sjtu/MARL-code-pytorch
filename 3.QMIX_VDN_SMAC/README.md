# QMIX and VDN in StarCraft II environment
This is a concise Pytorch implementation of QMIX and VDN in StarCraft II environment(SMAC-StarCraft Multi-Agent Challenge).<br />

## How to use my code?
You can dircetly run 'QMIX_SMAC_main.py' in your own IDE.<br />
If you want to use QMIX, you can set the paraemeter 'algorithm' = 'QMIX';<br />
If you want to use VDN, you can set the paraemeter 'algorithm' = 'VDN'.<br />

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
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/QMIX_SMAC_training_result.png)

## Reference
[1] Rashid T, Samvelyan M, Schroeder C, et al. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2018: 4295-4304.<br />
[2] Sunehag P, Lever G, Gruslys A, et al. Value-decomposition networks for cooperative multi-agent learning[J]. arXiv preprint arXiv:1706.05296, 2017.<br />
[3] [EPyMARL](https://github.com/uoe-agents/epymarl).<br />
[4] https://github.com/starry-sky6688/StarCraft.<br />
