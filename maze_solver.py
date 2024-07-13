import numpy as np

class MDP:
    def __init__(self, states, actions, transition_probs, rewards, gamma=0.99):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma

    def value_iteration(self, epsilon=0.001):
        V = np.zeros(len(self.states))
        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                V[s] = max(sum(p * (r + self.gamma * V[s_])
                              for p, s_, r in self.transition_probs[s][a])
                              for a in self.actions)
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        return V

    def policy_iteration(self):
        policy = np.zeros(len(self.states), dtype=int)
        V = np.zeros(len(self.states))
        while True:
            # Policy evaluation
            while True:
                delta = 0
                for s in self.states:
                    v = V[s]
                    a = policy[s]
                    V[s] = sum(p * (r + self.gamma * V[s_])
                               for p, s_, r in self.transition_probs[s][a])
                    delta = max(delta, abs(v - V[s]))
                if delta < 1e-6:
                    break

            # Policy improvement
            policy_stable = True
            for s in self.states:
                old_action = policy[s]
                policy[s] = np.argmax([sum(p * (r + self.gamma * V[s_])
                                           for p, s_, r in self.transition_probs[s][a])
                                           for a in self.actions])
                if old_action != policy[s]:
                    policy_stable = False
            if policy_stable:
                break
        return policy, V

# Example of how to use the MDP class
states = range(16)  # 4x4 grid
actions = ['up', 'down', 'left', 'right']
transition_probs = {}  # Populate this with your transition probabilities
rewards = {}  # Populate this with your rewards

mdp = MDP(states, actions, transition_probs, rewards)
optimal_value = mdp.value_iteration()
optimal_policy, optimal_value = mdp.policy_iteration()
