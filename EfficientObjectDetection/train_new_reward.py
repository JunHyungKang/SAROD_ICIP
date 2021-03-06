import os
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import deque
import pickle
import pylab
cudnn.benchmark = True
import argparse
from torch.autograd import Variable
# from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli
from collections import deque
import random

from EfficientObjectDetection.utils import utils_ete, utils_detector
from EfficientObjectDetection.constants import base_dir_metric_cd, base_dir_metric_fd
from EfficientObjectDetection.constants import num_actions
import yolov5.utils.utils as yoloutil

import warnings
warnings.simplefilter("ignore")


class EfficientOD():
    def __init__(self, opt):
        # GPU Device
        self.opt = opt
        self.result_fine = None
        self.result_coarse = None
        self.epoch = None
        gpu_id = self.opt.gpu_id
        self.buffer = deque(maxlen=20000)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        use_cuda = torch.cuda.is_available()
        print("GPU device for EfficientOD: ", use_cuda)

        if not os.path.exists(self.opt.cv_dir):
            print(self.opt.cv_dir)
            os.makedirs(self.opt.cv_dir)
        # utils_ete.save_args(__file__, self.opt)

        self.agent = utils_ete.get_model(num_actions)
        self.critic = utils_ete.critic_model(1)

        # ---- Load the pre-trained model ----------------------
        if self.opt.load is not None:
            path = os.path.join('weights', self.opt.load)
            checkpoint = torch.load(path)
            self.agent.load_state_dict(checkpoint['agent'])
            print('loaded agent from %s' % opt.load)

        # Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
        if self.opt.parallel:
            agent = nn.DataParallel(self.agent)
        self.agent.cuda()
        self.critic.cuda()

        # Update the parameters of the policy network
        self.optimizer_agent = optim.Adam(self.agent.parameters(), lr=self.opt.lr)
        self.optimizer_critic = optim.Adam(self.agent.parameters(), lr=self.opt.lr)

    def train(self, epoch, result_fine, result_coarse):
        # Start training and testing
        self.epoch = epoch
        self.result_fine = result_fine
        self.result_coarse = result_coarse

        trainset = utils_ete.get_dataset(self.opt.img_size, self.result_fine, self.result_coarse, 'train')
        trainloader = torchdata.DataLoader(trainset, batch_size=self.opt.batch_size, shuffle=True,
                                           num_workers=self.opt.num_workers)

        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        for epoch in range(self.epoch, self.epoch + 1):
            self.agent.train()
            rewards, rewards_baseline, policies, stats_list, efficiency = [], [], [], [], []
            for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

                f_ap = targets['f_ap']
                f_ap = torch.cat(f_ap, dim=0).view([-1, 4])

                c_ap = targets['c_ap']
                c_ap = torch.cat(c_ap, dim=0).view([-1, 4])

                f_stats = targets['f_stats']
                c_stats = targets['c_stats']

                f_ob = targets['f_ob']
                f_ob = torch.cat(f_ob, dim=0).view([-1, 4])
                c_ob = targets['c_ob']
                c_ob = torch.cat(c_ob, dim=0).view([-1, 4])

                self.buffer.append([inputs.numpy(), f_ap, c_ap, f_stats, c_stats, f_ob, c_ob])

            pbar = tqdm.tqdm(range((epoch+1)*6))
            for i in pbar:
                # if len(self.buffer)>= 100:
                minibatch = random.sample(self.buffer, self.opt.step_batch_size)
                # else:
                # continue

                minibatch = np.array(minibatch)

                inputs, f_ap, c_ap = minibatch[:, 0].tolist(), minibatch[:, 1], minibatch[:, 2]
                f_stats, c_stats = minibatch[:, -4], minibatch[:, -3]
                f_ob, c_ob = minibatch[:, -2], minibatch[:, -1]

                # inputs = Variable(inputs)
                # if not self.opt.parallel:
                inputs = torch.tensor(inputs).squeeze().cuda()
                # Actions by the Agent
                probs = F.sigmoid(self.agent.forward(inputs))
                alpha_hp = np.clip(self.opt.alpha + epoch * 0.001, 0.6, 0.95)
                probs = probs * alpha_hp + (1 - alpha_hp) * (1 - probs)

                # Sample the policies from the Bernoulli distribution characterized by agent
                distr = Bernoulli(probs)
                policy_sample = distr.sample()

                # Test time policy - used as baseline policy in the training step
                policy_map = probs.data.clone()
                policy_map[policy_map < 0.5] = 0.0
                policy_map[policy_map >= 0.5] = 1.0
                policy_map = Variable(policy_map)

                # Get the batch wise metrics
                # f_p, c_p, f_r, c_r, f_ap, c_ap, f_loss, c_loss, f_ob, c_ob, f_stats, c_stats

                # Find the reward for baseline and sampled policy
                f_ap = [np.concatenate(x, axis=0) for x in zip(*f_ap)]
                c_ap = [np.concatenate(x, axis=0) for x in zip(*c_ap)]
                f_ob = [np.concatenate(x, axis=0) for x in zip(*f_ob)]
                c_ob = [np.concatenate(x, axis=0) for x in zip(*c_ob)]

                f_ap = torch.from_numpy(f_ap[0].reshape((-1, 4)))
                c_ap = torch.from_numpy(c_ap[0].reshape((-1, 4)))
                f_ob = torch.from_numpy(f_ob[0].reshape((-1, 4)))
                c_ob = torch.from_numpy(c_ob[0].reshape((-1, 4)))
                f_ob = f_ob.float()
                c_ob = c_ob.float()

                reward_map = utils_ete.compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy_map.cpu().data, self.opt.beta, self.opt.sigma)
                reward_sample = utils_ete.compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy_sample.cpu().data, self.opt.beta,
                                                         self.opt.sigma)
                advantage = reward_sample.cuda().float() - reward_map.cuda().float()

                # Find the loss for only the policy network
                loss = distr.log_prob(policy_sample)
                loss = loss * Variable(advantage).expand_as(policy_sample)
                # loss = loss.expand_as(policy_sample)
                loss = loss.mean()
                # print('\1', sum(self.critic(inputs)))
                # print('\1', sum(reward_map))
                loss = loss + F.smooth_l1_loss(sum(self.critic(inputs)), sum(reward_map.cuda().float()))

                self.optimizer_agent.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                self.optimizer_agent.step()
                self.optimizer_critic.step()

                rewards.append(reward_sample.cpu())
                rewards_baseline.append(reward_map.cpu())
                policies.append(policy_sample.data.cpu())

                for batch in range(self.opt.step_batch_size):
                    for ind, policy_element in enumerate(policy_sample.cpu().data[batch]):
                        efficiency.append(policy_element)
                        if i == 0:
                            for stats in c_stats[batch][ind]:
                                stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0), torch.squeeze(stats[2], 0), stats[3]))
                        elif i == 1:
                            for stats in f_stats[batch][ind]:
                                stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0), torch.squeeze(stats[2], 0), stats[3]))

            cal_stats_list = [np.concatenate(x, 0) for x in zip(*stats_list)]
            if len(cal_stats_list) and cal_stats_list[0].any():
                p, r, ap, f1, ap_class = yoloutil.ap_per_class(*cal_stats_list)
                p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                # pbar.set_description(('\n{} Epoch {} Step - RL Train AP: {} '.format(epoch, i, map50)))

            reward, sparsity, variance, policy_set = utils_ete.performance_stats(policies, rewards)

            print('\n{} Epoch - RL Train mean AP: {} / Efficiency: {} '.format(epoch, map50, sum(efficiency)/len(efficiency)))
            print('Train: %d | Rw: %.6f | S: %.3f | V: %.3f | #: %d' % (epoch, reward, sparsity, variance, len(policy_set)))

            result = epoch, reward.cpu().item(), sparsity.cpu().item(), variance.cpu().item(), map50, sum(efficiency)/len(efficiency)
            with open(self.opt.cv_dir+'/rl_train.txt', 'a') as f:
                f.write(str(result) + '\n')

            # save the model --- agent
            agent_state_dict = self.agent.module.state_dict() if self.opt.parallel else self.agent.state_dict()
            state = {
                'agent': agent_state_dict,
                'epoch': self.epoch,
                'reward': reward,
            }
            if self.epoch % 10 == 0:
                torch.save(state, self.opt.cv_dir + '/ckpt_E_{}'.format(self.epoch))

    def eval(self, epoch, test_fine, test_coarse):

        self.test_fine = test_fine
        self.test_coarse = test_coarse

        self.agent.eval()

        testset = utils_ete.get_dataset(self.opt.img_size, self.test_fine, self.test_coarse, 'eval')
        testloader = torchdata.DataLoader(testset, batch_size=self.opt.batch_size, shuffle=True,
                                          num_workers=self.opt.num_workers)

        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        rewards, metrics, policies, set_labels, stats_list, efficiency = [], [], [], [], [], []
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs = Variable(inputs, volatile=True)
            # if not self.opt.parallel:
            inputs = torch.tensor(inputs).cuda()

            # Actions by the Policy Network
            probs = F.sigmoid(self.agent(inputs))

            # Sample the policy from the agents output
            policy = probs.data.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0
            policy = Variable(policy)

            # offset_fd, offset_cd = utils_ete.read_offsets(targets, num_actions)
            # f_p, c_p, f_r, c_r, f_ap, c_ap, f_loss, c_loss, f_ob, c_ob
            f_ap = targets['f_ap']
            f_ap = torch.cat(f_ap, dim=0).view([-1, 4])

            c_ap = targets['c_ap']
            c_ap = torch.cat(c_ap, dim=0).view([-1, 4])

            f_stats = targets['f_stats']
            c_stats = targets['c_stats']

            f_ob = targets['f_ob']
            f_ob = torch.cat(f_ob, dim=0).view([-1, 4])
            c_ob = targets['c_ob']
            c_ob = torch.cat(c_ob, dim=0).view([-1, 4])

            f_ob = f_ob.float()
            c_ob = c_ob.float()

            reward = utils_ete.compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy.data.cpu().data, self.opt.beta, self.opt.sigma)
            # metrics, set_labels = utils_ete.get_detected_boxes(policy, targets, metrics, set_labels)

            rewards.append(reward)
            policies.append(policy.data)

            for ind, i in enumerate(policy.cpu().data[0]):
                efficiency.append(i)
                if i == 0:
                    for stats in c_stats[ind]:
                        stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0),
                                           torch.squeeze(stats[2], 0), stats[3]))
                elif i == 1:
                    for stats in f_stats[ind]:
                        stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0),
                                           torch.squeeze(stats[2], 0), stats[3]))


        reward, sparsity, variance, policy_set = utils_ete.performance_stats(policies, rewards)

        cal_stats_list = [np.concatenate(x, 0) for x in zip(*stats_list)]
        if len(cal_stats_list) and cal_stats_list[0].any():
            p, r, ap, f1, ap_class = yoloutil.ap_per_class(*cal_stats_list)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        print('{} Epoch - RL Eval AP: {} / Efficiency: {} '.format(epoch, map50, sum(efficiency)/len(efficiency)))
        print('RL Eval - Rw: %.4f | S: %.3f | V: %.3f | #: %d\n' % (reward, sparsity, variance, len(policy_set)))

        result = epoch, reward.cpu().item(), sparsity.cpu().item(), variance.cpu().item(), map50, sum(efficiency)/len(efficiency)
        with open(self.opt.cv_dir+'/rl_eval.txt', 'a') as f:
            f.write(str(result)+'\n')

        # # save the model --- agent
        # agent_state_dict = self.agent.module.state_dict() if self.opt.parallel else self.agent.state_dict()
        # state = {
        #   'agent': agent_state_dict,
        #   'epoch': self.epoch,
        #   'reward': reward,
        # }
        # if self.epoch % 5 == 0:
        #     torch.save(state, self.opt.cv_dir+'/ckpt_E_%d_R_%.2E'%(self.epoch, reward))

    def test(self, epoch, test_fine, test_coarse):

        self.test_fine = test_fine
        self.test_coarse = test_coarse

        self.agent.eval()

        testset = utils_ete.get_dataset(self.opt.img_size, self.test_fine, self.test_coarse, 'eval')
        testloader = torchdata.DataLoader(testset, batch_size=self.opt.batch_size, shuffle=True,
                                          num_workers=self.opt.num_workers)

        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        rewards, metrics, policies, set_labels, stats_list, efficiency = [], [], [], [], [], []
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs = Variable(inputs, volatile=True)
            # if not self.opt.parallel:
            inputs = torch.tensor(inputs).cuda()

            # Actions by the Policy Network
            probs = F.sigmoid(self.agent(inputs))

            # Sample the policy from the agents output
            policy = probs.data.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0
            policy = Variable(policy)

            # f_p, c_p, f_r, c_r, f_ap, c_ap, f_loss, c_loss, f_ob, c_ob
            f_ap = targets['f_ap']
            f_ap = torch.cat(f_ap, dim=0).view([-1, 4])

            c_ap = targets['c_ap']
            c_ap = torch.cat(c_ap, dim=0).view([-1, 4])

            f_stats = targets['f_stats']
            c_stats = targets['c_stats']

            f_ob = targets['f_ob']
            f_ob = torch.cat(f_ob, dim=0).view([-1, 4])
            c_ob = targets['c_ob']
            c_ob = torch.cat(c_ob, dim=0).view([-1, 4])

            f_ob = f_ob.float()
            c_ob = c_ob.float()

            reward = utils_ete.compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy.data.cpu().data, self.opt.beta, self.opt.sigma)
            # metrics, set_labels = utils_ete.get_detected_boxes(policy, targets, metrics, set_labels)

            rewards.append(reward)
            policies.append(policy.data)

            for ind, i in enumerate(policy.cpu().data[0]):
                efficiency.append(i)
                if i == 0:
                    for stats in c_stats[ind]:
                        stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0),
                                           torch.squeeze(stats[2], 0), stats[3]))
                elif i == 1:
                    for stats in f_stats[ind]:
                        stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0),
                                           torch.squeeze(stats[2], 0), stats[3]))

        reward, sparsity, variance, policy_set = utils_ete.performance_stats(policies, rewards)

        cal_stats_list = [np.concatenate(x, 0) for x in zip(*stats_list)]
        if len(cal_stats_list) and cal_stats_list[0].any():
            p, r, ap, f1, ap_class = yoloutil.ap_per_class(*cal_stats_list)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        print('{} Epoch - RL Test AP: {} / Efficiency: {} '.format(epoch, map50, sum(efficiency)/len(efficiency)))
        print('RL Test - Rw: %.4f | S: %.3f | V: %.3f | #: %d\n' % (reward, sparsity, variance, len(policy_set)))

        result = epoch, reward.cpu().item(), sparsity.cpu().item(), variance.cpu().item(), map50, sum(efficiency)/len(efficiency)
        with open(self.opt.cv_dir+'/rl_test.txt', 'a') as f:
            f.write(str(result)+'\n')

    def test_wip(self, fine_detector, coarse_detector):

        self.agent.eval()

        testset = utils_ete.get_dataset_test(self.opt.img_size, img_path=self.opt.test_path)
        testloader = torchdata.DataLoader(testset, batch_size=self.opt.batch_size, shuffle=True,
                                          num_workers=self.opt.num_workers)

        p, r, f1, mp, mr, map50, map, t0, t1, c_map50, f_map50 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        stats_list, metrics, policies, set_labels, c_stats_list, f_stats_list, efficiency = [], [], [], [], [], [], []
        for batch_idx, (inputs, label_path) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs = Variable(inputs, volatile=True)
            # if not self.opt.parallel:
            inputs = torch.tensor(inputs).cuda()

            # Actions by the Policy Network
            probs = F.sigmoid(self.agent(inputs))

            # Sample the policy from the agents output
            policy = probs.data.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0
            policy = Variable(policy)

            for ind, i in enumerate(policy.cpu().data[0]):
                efficiency.append(i)
                if i == 0:
                    c_stats = coarse_detector.test(inputs, label_path, ind)
                    for c_stat in c_stats:
                        # stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0),
                        #                    torch.squeeze(stats[2], 0), stats[3]))
                        c_stats_list.append(c_stat)
                        stats_list.append(c_stat)
                elif i == 1:
                    f_stats = fine_detector.test(inputs, label_path, ind)
                    for f_stat in f_stats:
                        # stats_list.append((torch.squeeze(stats[0], 0), torch.squeeze(stats[1], 0),
                        #                    torch.squeeze(stats[2], 0), stats[3]))
                        f_stats_list.append(f_stat)
                        stats_list.append(f_stat)

            policies.append(policy.data)

        cal_stats_list = [np.concatenate(x, 0) for x in zip(*stats_list)]
        c_cal_stats_list = [np.concatenate(x, 0) for x in zip(*c_stats_list)]
        f_cal_stats_list = [np.concatenate(x, 0) for x in zip(*f_stats_list)]
        if len(cal_stats_list) and cal_stats_list[0].any():
            p, r, ap, f1, ap_class = yoloutil.ap_per_class(*cal_stats_list)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        if len(c_cal_stats_list) and c_cal_stats_list[0].any():
            c_p, c_r, c_ap, c_f1, c_ap_class = yoloutil.ap_per_class(*c_cal_stats_list)
            c_p, c_r, c_ap50, c_ap = c_p[:, 0], c_r[:, 0], c_ap[:, 0], c_ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            c_mp, c_mr, c_map50, c_map = c_p.mean(), c_r.mean(), c_ap50.mean(), c_ap.mean()
        if len(f_cal_stats_list) and f_cal_stats_list[0].any():
            f_p, f_r, f_ap, f_f1, f_ap_class = yoloutil.ap_per_class(*f_cal_stats_list)
            f_p, f_r, f_ap50, f_ap = f_p[:, 0], f_r[:, 0], f_ap[:, 0], f_ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            f_mp, f_mr, f_map50, f_map = f_p.mean(), f_r.mean(), f_ap50.mean(), f_ap.mean()

        # print('Fine Detector AP: {}'.format(f_map50))
        print('Coarse Detector AP: {} / Fine Detector AP: {}'.format(c_map50, f_map50))
        print('RL Test AP: {} / Efficiency: {} '.format(map50, sum(efficiency)/len(efficiency)))

    def visualization(self, fine_detector, coarse_detector):

        self.agent.eval()

        testset = utils_ete.get_dataset_test(self.opt.img_size, img_path=self.opt.test_path)
        testloader = torchdata.DataLoader(testset, batch_size=self.opt.batch_size, shuffle=True,
                                          num_workers=self.opt.num_workers)

        result = {}
        for batch_idx, (inputs, label_path) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs = Variable(inputs, volatile=True)
            # if not self.opt.parallel:

            inputs = torch.tensor(inputs).cuda()

            # Actions by the Policy Network
            probs = F.sigmoid(self.agent(inputs))

            # Sample the policy from the agents output
            policy = probs.data.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0
            policy = Variable(policy)

            for ind, i in enumerate(policy.cpu().data[0]):
                # print('policy', ind, i)
                if i == 0:
                    coarse_detector.test(inputs, label_path, ind, i)

                elif i == 1:
                    fine_detector.test(inputs, label_path, ind, i)
