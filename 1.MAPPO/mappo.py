import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


# Centralized critic, the input is the global state
class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_len, N, actor_input_dim), prob.shape(mini_batch_size, episode_len, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


# Centralized critic, the input is the global state
class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_len, N, critic_input_dim), value.shape=(mini_batch_size, episode_len, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_len = args.episode_len
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim = args.obs_dim + args.N
            self.critic_input_dim = args.state_dim + args.N
        else:
            self.actor_input_dim = args.obs_dim
            self.critic_input_dim = args.state_dim

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        if self.set_adam_eps:
            print("------set adam eps------")
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, obs_n, evaluate=False):
        with torch.no_grad():
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_input = torch.cat([obs_n, torch.eye(self.N)], dim=-1)  # actor_input.shape=(N, obs_dim+N)
            else:
                actor_input = obs_n  # actor_input.shape=(N,obs_dim)
            prob = self.actor(actor_input)  # prob.shape=(N,action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=1)
                return a_n.numpy()
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s):
        with torch.no_grad():
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            if self.add_agent_id:
                # Add an one-hot vector to represent the agent_id
                critic_input = torch.cat([s, torch.eye(self.N)], dim=-1)  # critic_input.shape=(N,state_dim+N)
            else:
                critic_input = s  # critic_input.shape=(N,state_dim)
            v_n = self.critic(critic_input)  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.numpy_to_tensor()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_len,N)
            for t in reversed(range(self.episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_len,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_len,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_len, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_len, N)
                """
                if self.use_rnn:
                    probs_now, values_now = self.get_probs_and_values_rnn(batch, index)
                else:
                    probs_now, values_now = self.get_probs_and_values_mlp(batch, index)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_len, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_len, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, episode_len, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index])  # ratios.shape=(mini_batch_size, episode_len, N)

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.optimizer_actor.step()

                critic_loss = F.mse_loss(v_target[index], values_now)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def get_probs_and_values_rnn(self, batch, index):
        probs_now, values_now = [], []
        # If use RNN, we need to reset the rnn_hidden of actor and critic.
        self.actor.rnn_hidden = None
        self.critic.rnn_hidden = None
        for t in range(self.episode_len):
            if self.add_agent_id:
                # agent_id_one_hot.shape=(mini_batch_size, N, N)
                agent_id_one_hot = torch.eye(self.N).expand(self.mini_batch_size, -1, -1)
                # actor_input.shape=(mini_batch_size*N, obs_dim+N)
                actor_input = torch.cat([batch["obs_n"][index, t], agent_id_one_hot], dim=-1).view(-1, self.actor_input_dim)
                # critic_input.shape=(mini_batch_size*N, state_dim+N)
                critic_input = torch.cat([batch['s'][index, t].unsqueeze(1).repeat(1, self.N, 1), agent_id_one_hot], dim=-1).view(-1, self.critic_input_dim)
            else:
                # actor_input.shape=(mini_batch_size*N, obs_dim)
                actor_input = batch["obs_n"][index, t].view(-1, self.actor_input_dim)
                # critic_input.shape=(mini_batch_size*N, state_dim)
                critic_input = batch['s'][index, t].unsqueeze(1).repeat(1, self.N, 1).view(-1, self.critic_input_dim)

            prob = self.actor(actor_input)  # prob.shape=(mini_batch_size*N, action_dim)
            probs_now.append(prob.view(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
            v = self.critic(critic_input)  # v.shape=(mini_batch_size*N,1)
            values_now.append(v.view(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size，N)

        # Stack them according to the time (dim=1)
        probs_now = torch.stack(probs_now, dim=1)  # probs_now.shape=（mini_batch_size, episode_len, N, action_dim）
        values_now = torch.stack(values_now, dim=1)  # values_now.shape=(mini_batch_size, episode_len, N)
        return probs_now, values_now

    def get_probs_and_values_mlp(self, batch, index):
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).expand(self.mini_batch_size, self.episode_len, -1, -1)
            # actor_input.shape=(mini_batch_size, episode_len, N, obs_dim+N)
            actor_input = torch.cat([batch['obs_n'][index, :-1], agent_id_one_hot], dim=-1)
            # critic_input.shape=(mini_batch_size, episode_len, N, state_dim+N)
            critic_input = torch.cat([batch['s'][index, :-1].unsqueeze(2).repeat(1, 1, self.N, 1), agent_id_one_hot], dim=-1)
        else:
            # actor_input.shape=(mini_batch_size, episode_len, N, obs_dim)
            actor_input = batch['obs_n'][index, :-1]
            # critic_input.shape=(mini_batch_size, episode_len, N, state_dim)
            critic_input = batch['s'][index, :-1].unsqueeze(2).repeat(1, 1, self.N, 1)

        probs_now = self.actor(actor_input)  # probs_now.shape=(mini_batch_size, episode_len, N, action_dim)
        values_now = self.critic(critic_input).squeeze(-1)  # values_now.shape=(mini_batch_size, episode_len, N)
        return probs_now, values_now

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/{}/MAPPO_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/{}/MAPPO_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
