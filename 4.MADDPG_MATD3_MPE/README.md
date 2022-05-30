# MADDGP and MATD3 in MPE environment
This is a concise Pytorch implementation of MADDPG and MATD3 in MPE environment(Multi-Agent Particle-World Environment).<br />

## How to use my code?
You can dircetly run 'MADDPG_MATD3_main.py' in your own IDE.<br />

## Requirements
python==3.7.9<br />
numpy==1.19.4<br />
pytorch==1.5.0<br />
tensorboard==0.6.0<br />
gym==0.10.5<br />
[Multi-Agent Particle-World Environment(MPE)](https://github.com/openai/multiagent-particle-envs)

## Trainning environments
We train our MADDPG and MATD3 in 'simple_speaker_listener' and 'simple_spread' in MPE environment.<br />

## Trainning result
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/1.MAPPO_MPE/MAPPO_MPE_training_result.png)

## Reference
[1] Lowe R, Wu Y I, Tamar A, et al. Multi-agent actor-critic for mixed cooperative-competitive environments[J]. Advances in neural information processing systems, 2017, 30.<br />
[2] Ackermann J, Gabler V, Osa T, et al. Reducing overestimation bias in multi-agent domains using double centralized critics[J]. arXiv preprint arXiv:1910.01465, 2019.<br />
