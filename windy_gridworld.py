import numpy as np
import gym

def epsilon_greedy(Q, state, nA, epsilon=0.1):
    if np.random.rand() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(nA)

def sarsa(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
            if done:
                break
    return Q

def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state = env.reset()
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _ = env.step(action)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            if done:
                break
    return Q

env = gym.make('WindyGridworld-v0')
num_episodes = 1000
Q_sarsa = sarsa(env, num_episodes)
Q_q_learning = q_learning(env, num_episodes)
