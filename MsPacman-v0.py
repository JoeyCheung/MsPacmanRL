import gym
from RL_brain import DeepQNetwork

env = gym.make('MsPacman-v0') 
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n, n_features=2, learning_rate=0.01, e_greedy=0.9,replace_target_iter=300, memory_size=3000, e_greedy_increment=0.0002,) #Change parameter values

total_steps = 0

for i_episode in range(1000): #Change the range based off iterations needed to solve

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)

        reward =  #Change the reward

        RL.store_transition(observation, action, reward, observation_)
        
        ep_r += reward
        if total_steps > 1000:
            RL.learn()
        
        if done: #Change what happens when it's done (This might be fine)
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
