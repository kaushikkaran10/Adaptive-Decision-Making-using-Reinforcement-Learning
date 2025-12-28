import numpy as np


class QLearningAgent:
    """
    Q-learning agent used in my experiments on adaptive decision making.
    
    This agent is intentionally kept simple to study how exploration
    and reward feedback influence behavior in a non-stationary bandit
    setting.
    """
    
    def __init__(self, n_actions=3, learning_rate=0.1, epsilon=0.1, discount_factor=0.9, seed=None):
        """
        """
        """

        """
       
        """

        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        
        # Initialize Q-values to zero for all actions
        self.q_values = np.zeros(n_actions)
        
        # Track action counts for statistics
        self.action_counts = np.zeros(n_actions)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
    
    def select_action(self):
        """
        Selects an action using epsilon-greedy exploration.
        """

        # Epsilon-greedy action selection 
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.q_values)
        # Track action selection
        self.action_counts[action] += 1
        
        return action
    
    def update(self, action, reward):
        """
        Update the estimated value of the selected action.

        In the bandit setting, the update depends only on the
        immediate reward.
        """

        # Q-learning update rule for bandits
        td_error = reward - self.q_values[action]
        self.q_values[action] += self.learning_rate * td_error
    
    def get_best_action(self):
        """
        Get the action with the highest Q-value.
        
        Returns:
            int: Action with highest Q-value
        """
        return np.argmax(self.q_values)
    
    def get_q_values(self):
        """
        Get a copy of current Q-values.
        
        Returns:
            numpy.ndarray: Copy of Q-values
        """
        return self.q_values.copy()
    
    def get_action_counts(self):
        """
        Get a copy of action selection counts.
        
        Returns:
            numpy.ndarray: Copy of action counts
        """
        return self.action_counts.copy()
    
    def reset(self):
        """
        Reset the agent's Q-values and action counts.
        """
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

