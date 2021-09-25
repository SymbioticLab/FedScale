import torch
import logging
import math
from utils.nlp import mask_tokens
from torch.autograd import Variable
from optimizer import ClientOptimizer
from utils.dqn import *
from client import Client

class RLClient(Client):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        self.optimizer = ClientOptimizer()
        self.dqn = DQN(conf)
        pass

    def train(self, client_data, model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        device = conf.device
        model = model.to(device=device)
        # self.dqn.eval_net = self.dqn.eval_net.to(device=device)
        # self.dqn.target_net = self.dqn.target_net.to(device=device)
        global_model = None

        if conf.gradient_policy == 'prox':
            # could be move to optimizer
            global_model = [param.data.clone() for param in model.parameters()]

        trained_unique_samples = conf.local_steps * conf.batch_size
        self.dqn.target_net.load_state_dict(model.state_dict())
        completed_steps = 0
        epoch_train_loss = 1e-4
        error_type = None

        while completed_steps < conf.local_steps:
            try:
                s = client_data.env.reset()
                episode_reward_sum = 0
                while True:
                    a = self.dqn.choose_action(s)
                    s_, r, done, info = client_data.env.step(a)
                    x, x_dot, theta, theta_dot = s_
                    r1 = (client_data.env.x_threshold - abs(x)) / client_data.env.x_threshold - 0.8
                    r2 = (client_data.env.theta_threshold_radians - abs(theta)) / client_data.env.theta_threshold_radians - 0.5
                    new_r = r1 + r2
                    self.dqn.store_transition(s, a, new_r, s_)
                    episode_reward_sum += new_r
                    s = s_
                    if self.dqn.memory_counter > conf.memory_capacity:
                        loss = self.dqn.learn()
                        loss_list = [loss.tolist()]
                        loss = loss.mean()
                        temp_loss = sum([l**2 for l in loss_list])/float(len(loss_list))

                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * temp_loss

                        completed_steps += 1

                    if done:
                        # print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                        break    
            except Exception as ex:
                error_type = ex
                break
            
        model.load_state_dict(self.dqn.target_net.state_dict())
        model_param = [param.data.cpu().numpy() for param in model.parameters()]
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results


    def test(self, args, rank, model, device):
        model = model.to(device=device)
        self.dqn.target_net.load_state_dict(model.state_dict())
        self.dqn.set_eval_mode()
        env = gym.make('CartPole-v0').unwrapped
        reward_sum = 0
        test_loss = 0
        s = env.reset()
        while True:
            a = self.dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            new_r = r1 + r2
            self.dqn.store_transition(s, a, new_r, s_)
            reward_sum += new_r
            s = s_
            if self.dqn.memory_counter > args.memory_capacity:
                test_loss += self.dqn.learn()

            if done:
                break    
        logging.info('Rank {}: Test set: Average loss: {}, Reward: {}'
          .format(rank, test_loss, reward_sum))
        return 0, 0, 0, {'top_1':reward_sum, 'top_5':reward_sum, 'test_loss':test_loss, 'test_len':1}
        


