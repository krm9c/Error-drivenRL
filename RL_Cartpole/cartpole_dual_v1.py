import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time, copy
import random
from torch.autograd import Variable, grad
torch.manual_seed(0)
import math, random
import gym
from collections import namedtuple

import argparse
# Sanity Check
print(torch.cuda.is_available())
from collections import deque
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

    
env_id = "CartPole-v0"
env = gym.make(env_id)
epsilon_start = 0.9
epsilon_final = 0.05
epsilon_decay = 200
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) *\
math.exp(-1. * frame_idx / epsilon_decay)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )   
    def forward(self, x):
        return self.layers(x)
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action



def compute_td_loss(model, target, optimizer, batch_size, replay_buffer, r1,  gamma=0.999, K=100, opt='SGD'):
    state, action, reward,\
    next_state, done = replay_buffer.sample(batch_size)
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done)) 
    
    # A simpler way of doing the same things. 
    q_values      = model(state)
    next_q_values = model(next_state)
    next_q_state_values = target(next_state) 
    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    if opt=='SGD':
        # print(opt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
    else:
        # print(opt)
        for parameter in model.parameters():
            # Low rank stuff
            # if len(parameter.shape) > 1:
            #     gradient= grad(loss,parameter, retain_graph = True)[0]
            #     u,s,v = torch.svd(gradient, compute_uv = True)
            #     # Low rank SVD
            #     M, N = list(gradient.size())[0], list(gradient.size())[1]
            #     if 2*K > max(M, N):
            #         omega = torch.rand( size=(N, max(M,N) ) ) 
            #     else:
            #         omega = torch.rand(size=( N, 2*K))
            #     omega_pm = torch.mm(gradient,torch.transpose(gradient,0,1))
            #     Y = torch.mm(omega_pm,torch.mm(gradient,omega))
            #     Q, R = torch.qr(Y)
            #     B = torch.mm(torch.transpose(Q,0,1),gradient)
            #     u, s, v = torch.svd(B)
            #     u = torch.mm(Q,u)
            #     grad_modified = torch.mm(torch.mm(u, torch.diag(s+r1)), v.t())
            #     parameter.grad= torch.mm(torch.mm(u, torch.diag(s+r1)), v.t())
            # else:
            #     parameter.grad = grad(loss,parameter, create_graph = True)[0]   
            # optimizer.step()

            ## Non low rank stuff
            if parameter.grad is not None:
                parameter.grad.zero_()
            if len(parameter.shape) > 1:
                gradient = grad(loss,parameter, create_graph = True)[0]
                u,s,v = torch.svd(gradient, compute_uv = True)
                parameter.grad= torch.mm(torch.mm(u, torch.diag(s+r1)), v.t())
            else:
                parameter.grad = grad(loss,parameter, create_graph = True)[0]      
        optimizer.step() 
    return loss



## The main network function 
def run_network_double(p, exploration_rate = 0.0001,\
                buffer_size  = 1000, update_model = 100, \
                     file_name='store_rewards', flag='non_default', opt='SGD'):
    import sys
    num_frames = p['num_frames']
    batch_size = p['batch_size']
    gamma= p['gamma']

    losses = []
    all_rewards = []
    episode_reward = 0
    state = env.reset()
    flag = 0

    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model  = DQN(env.observation_space.shape[0], env.action_space.n)
    optimizer_1   = optim.Adam(current_model.parameters() ) 
    optimizer_2   = optim.Adam(target_model.parameters() )

    replay_buffer = ReplayBuffer(buffer_size)
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        if flag == 0:
            action = current_model.act(state, epsilon)
        else:
            action = target_model.act(state, epsilon)    
        
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
        
        if len(replay_buffer) > batch_size:
            if flag ==0:
                loss = compute_td_loss(current_model, target_model ,optimizer_1, \
                    batch_size, replay_buffer, r1 = exploration_rate, gamma=gamma, opt=opt)  
            elif flag==1:
                loss = compute_td_loss(target_model, current_model ,optimizer_2,\
                    batch_size, replay_buffer, r1 = exploration_rate, gamma=gamma, opt=opt)
            losses.append(loss.item())


        if frame_idx % update_model == 0:   
            # exploration_rate = exploration_rate*gamma
            if flag == 0:
                flag = 1
            else:
                flag = 0
            sys.stdout.write('\rProgress: %d/%d,\
                    Reward: %f,  Loss: %f' %(frame_idx, num_frames, np.mean(all_rewards[-10:]),\
                    np.mean(losses[-10:])) )
            sys.stdout.flush()
            

    print( len(all_rewards) )
    current_model = None
    target_model  = None
    optimizer_1   = None
    optimizer_2   = None
    replay_buffer = None

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,5))
    # plt.subplot(131)
    # plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
    # plt.plot(all_rewards)
    # plt.subplot(132)
    # plt.title('loss')
    # plt.yscale('log')
    # plt.plot(losses)
    # plt.savefig(file_name+'_rewards_'+str(buffer_size)+'_'+str(exploration_rate)+'_'+str(update_model)+'.png', dpi=600)
    # plt.show()
    # plt.close()

    if flag=='non_default':
        np.savetxt(file_name+'_'+str(buffer_size)+'_'+str(exploration_rate)+'_'+str(update_model)+'_.csv', all_rewards, delimiter = ',')
    else:
        np.savetxt(file_name+'_.csv', all_rewards, delimiter = ',')
    return 0





def update_target(current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())



## The main network function 
def run_network_single(p, exploration_rate = 0.0001,\
                buffer_size  = 1000, update_model = 100,\
                file_name='store_rewards', flag='non_default', opt='SGD'):
    
    num_frames=p['num_frames']
    batch_size=p['batch_size']
    gamma=p['gamma']
    
    import sys
    losses = []
    all_rewards = []
    all_loss =[]
    episode_reward = 0
    state = env.reset()
    flag = 0


    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model  = DQN(env.observation_space.shape[0], env.action_space.n)
    optimizer     = optim.Adam(current_model.parameters()) 

    replay_buffer = ReplayBuffer(buffer_size)
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)  
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward



    current.append( current_model(Variable(torch.FloatTensor(np.float32(state)))).mean().item() )
    target.append(   target_model(Variable(torch.FloatTensor(np.float32(state)))).mean().item() )




        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(current_model, target_model, optimizer, batch_size,\
                 replay_buffer, r1 = exploration_rate, gamma=gamma, opt=opt)   


            losses =  (np.array(current).reshape([-1]) - np.array(target).reshape([-1]))
            all_loss.append( np.square(losses.mean().item()) )
            losses.append(loss.item())


        if frame_idx % update_model == 0:   
            # exploration_rate = exploration_rate*gamma
            update_target(current_model, target_model)
            sys.stdout.write('\rProgress: %d/%d,\
                    Reward: %f,  Loss: %f,  Loss:(Q difference): %f' %(frame_idx, num_frames, np.mean(all_rewards[-50:]),\
                    np.mean(losses[-50:])), np.mean(all_loss[-50:]) )
            sys.stdout.flush()
        

    # if frame_idx%400 == 0:
    #      plot(frame_idx, all_rewards, losses) 
    current_model = None
    target_model  = None
    optimizer   = None
    replay_buffer = None


    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,5))
    # plt.subplot(131)
    # plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
    # plt.plot(all_rewards)
    # plt.subplot(132)
    # plt.title('loss')
    # plt.yscale('log')
    # plt.plot(losses)
    # plt.savefig(file_name+'_rewards.png', dpi=600)
    # plt.show()
    # plt.close()

    print(all_rewards)
    if flag=='non_default':
        np.savetxt(file_name+'_'+str(buffer_size)+'_'+str(exploration_rate)+'_'+str(update_model)+'_.csv', all_rewards, delimiter = ',')
    else:
        np.savetxt(file_name+'_.csv', all_rewards, delimiter = ',')
        np.savetxt(file_name+'losses_.csv', all_loss, delimiter = ',')

    return 0




################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_file',
    default=None,
    help="filename")

parser.add_argument(
    '--save_model',
    default=None,
    help="save_model")

parser.add_argument(
    '--version',
    default='double_EDL',
    help="single_EDL, double_EDL, single_grad, double_grad")

parser.add_argument(
    '--param',
    default='default',
    help="exp, buff, model_updates, all")

parser.add_argument(
    '--gamma',
    default=0.999,
    help="discount factor")

parser.add_argument(
    '--num_frames',
    default=10000,
    help="total number of frames for the model")


parser.add_argument(
    '--batch_size',
    default=32,
    help="batch size")



if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    params=dict()
    ##############################################
    if args.output_file is not None:
        params['output_file'] = args.output_file

    if args.save_model is not None:
        params['save_model'] = args.save_model

    if args.version is not None:
        params['version'] = args.version

    if args.param is not None:
        params['param'] = str(args.param)

    if args.gamma is not None:
        params['gamma'] = float(args.gamma)

    if args.num_frames is not None:
        params['num_frames'] = int(args.num_frames)

    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)


    if params['version'] == 'double_EDL':
        if params['param']== 'exp':
            ## The  Double model runs for EDL
            # #### Run the model with varying exploration rate
            print(params['version'], params['param'])
            for exp in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print("\n running network for exploration", exp)
                run_network_double(p=params, exploration_rate=exp, file_name= 'data_dual_NN_EDL/rewards_explo_Rate_')

        elif params['param']== 'buff':
            print(params['version'], params['param'])
            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:

                print("\n running network for buffer", exp)
                run_network_double(p=params, buffer_size=exp, file_name= 'data_dual_NN_EDL/rewards_buff_size_')

        elif params['param']== 'model_updates':
            print(params['version'], params['param'])
            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_double(p=params,update_model=exp, file_name= 'data_dual_NN_EDL/rewards_update_step_')

        elif params['param']== 'all':
            ## The  Double model runs for EDL
            # #### Run the model with varying exploration rate
            print(params['version'], params['param'])
            for exp in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print("\n running network for exploration", exp)
                run_network_double(p=params,exploration_rate=exp, file_name= 'data_dual_NN_EDL/rewards_explo_Rate_')

            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_double(p=params,buffer_size=exp, file_name= 'data_dual_NN_EDL/rewards_buff_size_')


            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_double(p=params,update_model=exp, file_name= 'data_dual_NN_EDL/rewards_update_step_')
        else: 
            run_network_double(p=params, file_name= 'data_dual_NN_EDL/rewards_dual_default_v2', opt='EDL')



    elif params['version'] == 'single_EDL':
        if params['param']== 'exp':
            print(params['version'], params['param'])
            ## The  Double model runs for EDL
            # #### Run the model with varying exploration rate
            for exp in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print("\n running network for exploration", exp)
                run_network_single(p=params, exploration_rate=exp, file_name= 'data_single_NN_EDL/rewards_explo_Rate_')

        elif params['param']== 'buff':
            print(params['version'], params['param'])
            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_single(p=params, buffer_size=exp, file_name= 'data_single_NN_EDL/rewards_buff_size_')
                

        elif params['param']== 'model_updates':
            print(params['version'], params['param'])
            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_single(p=params, update_model=exp, file_name= 'data_single_NN_EDL/rewards_update_step_')

        elif params['param']== 'all':
            print(params['version'], params['param'])
            ## The  Double model runs for EDL
            # #### Run the model with varying exploration rate
            for exp in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print("\n running network for exploration", exp)
                run_network_single(p=params, exploration_rate=exp, file_name= 'data_single_NN_EDL/rewards_explo_Rate_')

            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_single(p=params, buffer_size=exp, file_name= 'data_single_NN_EDL/rewards_buff_size_')

            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_single(p=params, update_model=exp, file_name= 'data_single_NN_EDL/rewards_update_step_')
        else: 
            print(params['version'], params['param'])
            run_network_single(p=params, file_name= 'data_single_NN_EDL/rewards_default_v2', opt='EDL')


        
    elif params['version'] == 'single_grad':
        print(params['version'], params['param'])
        if params['param']== 'buff':
            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_single(p=params, exploration_rate=0, buffer_size=exp, file_name= 'data_single_NN_SGD/rewards_buff_size_')

        elif params['param']== 'model_updates':
            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_single(p=params, exploration_rate=0, update_model=exp, file_name= 'data_single_NN_SGD/rewards_update_step_')

        elif params['param']== 'all':
            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_single(p=params, exploration_rate=0,  buffer_size=exp, file_name= 'data_single_NN_SGD/rewards_buff_size_')

            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_single(p=params, exploration_rate=0, update_model=exp, file_name= 'data_single_NN_SGD/rewards_update_step_')
        else: 
            print(params['version'], params['param'])
            run_network_single(p=params,  file_name= 'data_single_NN_SGD/rewards_default_v2', opt='SGD')



    elif params['version'] == 'double_grad':
        print(params['version'], params['param'])
        if params['param']== 'buff':
            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_double(p=params, exploration_rate=0, buffer_size=exp, file_name= 'data_dual_NN_SGD/rewards_buff_size_')
                break

        elif params['param']== 'model_updates':
            print(params['version'], params['param'])
            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_double(p=params, exploration_rate=0, update_model=exp, file_name= 'data_dual_NN_SGD/rewards_update_step_')

        elif params['param']== 'all':
            print(params['version'], params['param'])
            # ### Now run the model with varying buffer sizes
            for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                print("\n running network for buffer", exp)
                run_network_double(p=params, exploration_rate=0, buffer_size=exp, file_name= 'data_dual_NN_SGD/rewards_buff_size_')

            for exp in  [10, 20, 30, 50, 100, 150, 200]:
                print("\n running network for update steps", exp)
                run_network_double(p=params, exploration_rate=0, update_model=exp, file_name= 'data_dual_NN_SGD/rewards_update_step_')
        else: 
            print(params['version'], params['param'])
            run_network_double(p=params, file_name= 'data_dual_NN_SGD/rewards_default_v2', opt='SGD')





## The  Double model runs for EDL
# #### Run the model with varying exploration rate
# for exp in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     print("\n running network for exploration", exp)
#     run_network_double(exploration_rate=exp, file_name= 'data_dual_NN_EDL/rewards_dual_explo_Rate_')
#     break


# ### Now run the model with varying buffer sizes
# for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
#     print("\n running network for buffer", exp)
#     run_network_double(buffer_size=exp, file_name= 'data_dual_NN_EDL/rewards_dual_buff_size_')
#     break


# ### Now run the model with varying time step between model updates
# for exp in  [10, 20, 30, 50, 100, 150, 200]:
#     print("\n running network for update steps", exp)
#     run_network_double(update_model=exp, file_name= 'data_dual_NN_EDL/rewards_dual_update_step_')
#     break






# The  Double model runs for grad
## Now run the model with varying buffer sizes
# for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
#     print("\n running network for buffer", exp)
#     run_network_double(exploration_rate=0, buffer_size=exp, file_name= 'data_dual_NN_EDL/rewards_dual_buff_size_')
#     break


# ### Now run the model with varying time step between model updates
# for exp in  [10, 20, 30, 50, 100, 150, 200]:
#     print("\n running network for update steps", exp)
#     run_network_double(exploration_rate=0, update_model=exp, file_name= 'data_dual_NN_EDL/rewards_dual_update_step_')
#     break




# The single model runs for grad
### Now run the model with varying buffer sizes
# for exp in  [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
#     print("\n running network for buffer", exp)
#     run_network_single(exploration_rate=0, buffer_size=exp, file_name= 'data_dual_NN_SGD/rewards_dual_buff_size_')
#     break


# ### Now run the model with varying time step between model updates
# for exp in  [10, 20, 30, 50, 100, 150, 200]:
#     print("\n running network for update steps", exp)
#     run_network_single(exploration_rate=0, update_model=exp, file_name= 'data_dual_NN_SGD/rewards_dual_update_step_')
#     break