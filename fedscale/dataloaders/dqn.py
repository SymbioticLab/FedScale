import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RLData(object):
    def __init__(self, args):
        self.args = args
        self.env = gym.make('CartPole-v0').unwrapped

    def getSize(self):
        return {'size': [self.args.local_steps * self.args.batch_size for _ in range(self.args.num_participants)]}


class Net(nn.Module):
    def __init__(self, n_actions, n_states):
        super(Net, self).__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.fc1 = nn.Linear(self.n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, self.n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, args):
        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')
        self.eval_net, self.target_net = Net(args.n_actions, args.n_states).to(
            self.device), Net(args.n_actions, args.n_states).to(self.device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros(
            (self.args.memory_capacity, self.args.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=self.args.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        if np.random.uniform() < self.args.epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.args.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def set_eval_mode(self):
        self.eval_net.eval()
        self.target_net.eval()

    def set_train_mode(self):
        self.eval_net.train()
        self.target_net.train()

    def learn(self):
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(
            self.args.memory_capacity, self.args.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(
            b_memory[:, :self.args.n_states]).to(self.device)
        b_a = torch.LongTensor(
            b_memory[:, self.args.n_states:self.args.n_states+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(
            b_memory[:, self.args.n_states+1:self.args.n_states+2]).to(self.device)
        b_s_ = torch.FloatTensor(
            b_memory[:, -self.args.n_states:]).to(self.device)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.args.gamma * \
            q_next.max(1)[0].view(self.args.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
