import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from make_env import make_env
import argparse
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
from matd3 import MATD3
import copy


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Create env
        self.env = make_env(env_name, discrete=False)  # Continuous action space
        self.env_evaluate = make_env(env_name, discrete=False)
        self.args.N = self.env.n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space[i].shape[0] for i in range(self.args.N)]  # actions dimensions of N agents
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        else:
            print("Wrong!!!")

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
        self.evaluate_policy()

        while self.total_steps < self.args.max_train_steps:
            obs_n = self.env.reset()
            for _ in range(self.args.episode_limit):
                # Each agent selects actions based on its own local observations(add noise for exploration)
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                # --------------------------!!!注意！！！这里一定要deepcopy，MPE环境会把a_n乘5-------------------------------------------
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))
                # Store the transition
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.total_steps += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size:
                    # Train each agent individually
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()

                if all(done_n):
                    break

        self.env.close()
        self.env_evaluate.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            obs_n = self.env_evaluate.reset()
            episode_reward = 0
            for _ in range(self.args.episode_limit):
                a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
                obs_next_n, r_n, done_n, _ = self.env_evaluate.step(copy.deepcopy(a_n))
                episode_reward += r_n[0]
                obs_n = obs_next_n
                if all(done_n):
                    break
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward, self.noise_std))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        for agent_id in range(self.args.N):
            self.agent_n[agent_id].save_model(self.env_name, self.args.algorithm, self.number, self.total_steps, agent_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    env_names = ["simple_speaker_listener", "simple_spread"]
    env_index = 0
    runner = Runner(args, env_name=env_names[env_index], number=1, seed=0)
    runner.run()
