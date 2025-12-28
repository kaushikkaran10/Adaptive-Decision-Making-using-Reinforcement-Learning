import numpy as np


class NonStationaryBandit:
    
    def __init__(self, n_steps=1000, seed=None):
        self.n_actions = 3
        self.n_steps = n_steps
        self.current_step = 0
        self.change_point = n_steps // 2  # Halfway point where probabilities change
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Reward probabilities before and after the change point
        self.initial_probs = np.array([0.3, 0.6, 0.4])
        self.changed_probs = np.array([0.7, 0.3, 0.5])
        self.current_probs = self.initial_probs.copy()
    
    def step(self, action):
        # Validate action
        if not isinstance(action, (int, np.integer)) or action < 0 or action > 2:
            raise ValueError(f"Action must be 0, 1, or 2. Got {action}")
        
        # Check if we've reached the change point and update probabilities
        if self.current_step == self.change_point:
            self.current_probs = self.changed_probs.copy()
            print(f"\n[Environment Change] Step {self.current_step}: Reward probabilities have changed!")
            print(f"New probabilities: {self.current_probs}\n")
        
        # Sample reward from Bernoulli distribution based on action's probability
        reward = np.random.binomial(1, self.current_probs[action])
        
        # Increment step counter
        self.current_step += 1
        
        # Check if experiment is done
        done = self.current_step >= self.n_steps
        
        return reward, done
    
    def reset(self):
        self.current_step = 0
        self.current_probs = self.initial_probs.copy()
        
        return {
            'step': self.current_step,
            'current_probs': self.current_probs.copy()
        }
    
    def get_optimal_action(self):
        """
        Get the current optimal action (highest reward probability).
        
        Returns:
            int: The action with the highest reward probability
        """
        return np.argmax(self.current_probs)
    
    def get_state_info(self):
        return {
            'step': self.current_step,
            'current_probs': self.current_probs.copy(),
            'optimal_action': self.get_optimal_action(),
            'phase': 'initial' if self.current_step < self.change_point else 'changed'
        }


