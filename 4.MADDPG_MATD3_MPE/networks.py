import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim_n[agent_id])
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))

        return a


class Critic_MADDPG(nn.Module):
    def __init__(self, args):
        super(Critic_MADDPG, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q = F.relu(self.fc1(s_a))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        self.fc4 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1
