# MARL-code-pytorch
Concise pytorch implements of MARL algorithms, including MAPPO, MADDPG, MATD3, QMIX and VDN.

# Requirements
python==3.7.9<br />
numpy==1.19.4<br />
pytorch==1.5.0<br />
tensorboard==0.6.0<br />
gym==0.10.5<br />
[Multi-Agent Particle-World Environment(MPE)](https://github.com/openai/multiagent-particle-envs)<br />
[SMAC-StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)<br />

# Trainning results
## 1. MAPPO in MPE (discrete action space)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/1.MAPPO_MPE/MAPPO_MPE_training_result.png)

## 2. MAPPO in  StarCraft II(SMAC)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/2.MAPPO_SMAC/MAPPO_SMAC_training_result.png)

## 3. QMIX and VDN in StarCraft II(SMAC)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/QMIX_SMAC_training_result.png)

## 4. MADDPG and MATD3 in MPE (continuous action space)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/4.MADDPG_MATD3_MPE/MADDPG_MATD3_training_result.png)

# Some Details
In order to facilitate switching between discrete action space and continuous action space in MPE environments, we make some small modifications in [MPE source code](https://github.com/openai/multiagent-particle-envs).<br />
 ## 1. make_env.py
 We add an argument named 'discrete' in 'make_env.py',which is a bool variable.
 ![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/MPE%20make_env%20modification.png)
 ## 2. environment.py
 We also add an argument named 'discrete' in 'environment.py'.
 ![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/MPE%20environment%20modification.png)
 ## 3. How to create a MPE environment?
 If your want to use discrete action space mode, you can use 'env=make_env(scenario_name, discrete=True)' <br />
 If your want to use continuous action space mode, you can use 'env=make_env(scenario_name, discrete=False)' <br />

