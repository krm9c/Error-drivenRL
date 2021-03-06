from core.argparser import parse_args
from core.helpers import (compute_td_loss,
                          initialize_models,
                          is_atari,
                          set_device,
                          update_epsilon)
from core.replay_buffer import ReplayBuffer
from core.train_information import TrainInformation
from core.wrappers import wrap_environment

from torch import save
from torch.optim import Adam


def update_graph(model, target_model, optimizer_current, optimizer_target, replay_buffer, args, device,
                 info, flag):
    if len(replay_buffer) > args.initial_learning:
        if not info.index % args.target_update_frequency:
            flag = not flag
        batch = replay_buffer.sample(args.batch_size)
        if flag:
            compute_td_loss(target_model, model, batch, args.gamma, device)
            optimizer_target.step()
        else:
            compute_td_loss(model, target_model, batch, args.gamma, device)
            optimizer_current.step()
    return flag


def complete_episode(model, environment, info, episode_reward, episode,
                     epsilon):
    new_best = info.update_rewards(episode_reward)
    if new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
        save(model.state_dict(), '%s.dat' % str('Res_model/')+environment)
    print('Episode %s - Reward: %s, Best: %s, Average: %s '
          'Epsilon: %s' % (episode, episode_reward, info.best_reward,
                           round(info.average, 3), round(epsilon, 4)))


def run_episode(env, model, target_model, optimizer_current, optimizer_target, replay_buffer, args,
                device, info, episode):
    episode_reward = 0.0
    state = env.reset()
    flag = False
    while True:
        epsilon = update_epsilon(info.index, args)
        action = model.act(state, epsilon, device)
        if args.render:
            env.render()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        info.update_index()
        flag = update_graph(model, target_model, optimizer_current, optimizer_target, replay_buffer, args,
                     device, info, flag)
        if done:
            complete_episode(model, args.environment, info, episode_reward,
                             episode, epsilon)
            break


def train(env, model, target_model, optimizer_current, optimizer_target, replay_buffer, args, device):
    info = TrainInformation()
    for episode in range(args.num_episodes):
        run_episode(env, model, target_model, optimizer_current, optimizer_target, replay_buffer, args,
                    device, info, episode)


def main():
    args = parse_args()
    env = wrap_environment(args.environment)
    device = set_device(args.force_cpu)
    model, target_model = initialize_models(env, device)

    optimizer_current = Adam(model.parameters(),        lr=args.learning_rate)
    optimizer_target  = Adam(target_model.parameters(), lr=args.learning_rate)


    replay_buffer = ReplayBuffer(args.buffer_capacity)
    train(env, model, target_model, optimizer_current, optimizer_target, replay_buffer, args, device)
    env.close()


if __name__ == '__main__':
    main()
