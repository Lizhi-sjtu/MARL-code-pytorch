import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from make_env import make_env
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo import MAPPO


def evaluate_policy(env, agent_n, state_norm, args):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        obs_n = env.reset()
        if args.use_state_norm:
            obs_n = state_norm(obs_n, update=False)
        if args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of actor and critic.
            agent_n.actor.rnn_hidden = None
        episode_reward = 0
        for _ in range(args.episode_len):
            a_n = agent_n.choose_action(obs_n, evaluate=True)
            obs_next_n, r_n, done_n, _ = env.step(a_n)
            if args.use_state_norm:
                obs_next_n = state_norm(obs_next_n, update=False)
            obs_n = obs_next_n
            episode_reward += r_n[0]

        evaluate_reward += episode_reward
    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = make_env(env_name, discrete=True)  # Discrete action space
    env_evaluate = make_env(env_name, discrete=True)
    args.N = env.n  # The number of agents
    args.obs_dim_n = [env.observation_space[i].shape[0] for i in range(args.N)]  # obs dimensions of N agents
    args.action_dim_n = [env.action_space[i].n for i in range(args.N)]  # actions dimensions of N agents
    # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
    args.obs_dim = args.obs_dim_n[0]  # The dimensions of an agent's observation space
    args.action_dim = args.action_dim_n[0]  # The dimensions of an agent's action space
    args.state_dim = np.sum(args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
    print("observation_space=", env.observation_space)
    print("obs_dim_n={}".format(args.obs_dim_n))
    print("action_space=", env.action_space)
    print("action_dim_n={}".format(args.action_dim_n))

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    episode = 0

    replay_buffer = ReplayBuffer(args)
    agent_n = MAPPO(args)  # All agents share the actor and critic

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=(args.N, args.obs_dim))
    if args.use_reward_norm:
        print("use reward norm")
        reward_norm = Normalization(shape=args.N)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=args.N, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        obs_n = env.reset()  # Note: obs_n=[obs of agent_1, obs of agent_2,..., obs of agent_N]
        if args.use_state_norm:
            obs_n = state_norm(obs_n)
        if args.use_reward_scaling:
            reward_scaling.reset()
        if args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of actor and critic.
            agent_n.actor.rnn_hidden = None
            agent_n.critic.rnn_hidden = None
        episode_obs_n, episode_s, episode_v_n, episode_a_n, episode_a_logprob_n, episode_r_n, episode_done_n = [], [], [], [], [], [], []
        for _ in range(args.episode_len):
            a_n, a_logprob_n = agent_n.choose_action(obs_n)  # Get actions and the corresponding log probabilities of N agents
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _ = env.step(a_n)

            if args.use_state_norm:
                obs_next_n = state_norm(obs_next_n)
            if args.use_reward_norm:
                r_n = reward_norm(r_n)
            elif args.use_reward_scaling:
                r_n = reward_scaling(r_n)
            # store the transition
            episode_obs_n.append(obs_n)
            episode_s.append(s)
            episode_v_n.append(v_n)
            episode_a_n.append(a_n)
            episode_a_logprob_n.append(a_logprob_n)
            episode_r_n.append(r_n)
            episode_done_n.append(done_n)

            obs_n = obs_next_n
            total_steps += 1
            if all(done_n):
                break

        # store the last obs、s and v
        s = np.array(obs_next_n).flatten()
        v_n = agent_n.get_value(s)
        episode_obs_n.append(obs_next_n)
        episode_s.append(s)
        episode_v_n.append(v_n)
        episode += 1
        # Store the transitions of an episode
        replay_buffer.store_episode(episode_obs_n, episode_s, episode_v_n, episode_a_n, episode_a_logprob_n, episode_r_n, episode_done_n)
        # When the number of episodes in replay buffer reaches batch_size,then update
        if replay_buffer.count == args.batch_size:
            agent_n.train(replay_buffer, total_steps)
            replay_buffer.count = 0

        if episode % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(env_evaluate, agent_n, state_norm, args)
            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
            writer.add_scalar('evaluate_step_rewards_{}'.format(env_name), evaluate_reward, global_step=total_steps)

            # Save the rewards and models
            if evaluate_num % args.save_freq == 0:
                np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
                agent_n.save_model(env_name, number, seed, total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_len", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=40, help="Save frequency")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization. Here, we do not use it.")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN, in MPE, we do not need to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")

    args = parser.parse_args()
    main(args, env_name="simple_spread", number=1, seed=0)
