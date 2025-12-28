import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from .environment import NonStationaryBandit
from .agent import QLearningAgent


def run_experiment(n_steps=1000, learning_rate=0.1, epsilon=0.1, seed=42):
    # Create environment and agent
    env = NonStationaryBandit(n_steps=n_steps, seed=seed)
    agent = QLearningAgent(n_actions=3, learning_rate=learning_rate, epsilon=epsilon, seed=seed)
    
    # Storage for results
    rewards = []
    cumulative_rewards = []
    actions_taken = []
    q_values_history = []
    
    # Track actions before and after environment change
    change_point = env.change_point
    actions_before_change = []
    actions_after_change = []

    # Reset environment
    env.reset()
    total_reward = 0
    
    # Run experiment
    for step in range(n_steps):
        # Agent selects action
        action = agent.select_action()
        
        # Environment returns reward
        reward, done = env.step(action)
        
        # Agent updates Q-values
        agent.update(action, reward)
        
        # Store results
        rewards.append(reward)
        total_reward += reward
        cumulative_rewards.append(total_reward)
        actions_taken.append(action)
        q_values_history.append(agent.get_q_values())
        
        # Track actions before/after change
        if step < change_point:
            actions_before_change.append(action)
        else:
            actions_after_change.append(action)
        
        # Print progress at key points
        if step in [0, change_point - 1, change_point, n_steps - 1]:
            print(f"Step {step:4d}: Action={action}, Reward={reward}, "
                  f"Q-values={agent.q_values.round(3)}, "
                  f"Cumulative Reward={total_reward}")
        
        if done:
            break
    
    # Return results
    return {
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
        'actions_taken': actions_taken,
        'q_values_history': np.array(q_values_history),
        'actions_before_change': actions_before_change,
        'actions_after_change': actions_after_change,
        'change_point': change_point,
        'total_reward': total_reward,
        'agent': agent,
        'env': env
    }


def plot_results(results):
    rewards = results['rewards']
    cumulative_rewards = results['cumulative_rewards']
    actions_taken = results['actions_taken']
    q_values_history = results['q_values_history']
    change_point = results['change_point']
    actions_before_change = results['actions_before_change']
    actions_after_change = results['actions_after_change']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning on Non-Stationary Multi-Armed Bandit', fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative Reward over Time
    ax1 = axes[0, 0]
    ax1.plot(cumulative_rewards, linewidth=2, color='#2E86AB')
    ax1.axvline(x=change_point, color='red', linestyle='--', linewidth=2, label='Environment Change')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Cumulative Reward', fontsize=12)
    ax1.set_title('Cumulative Reward Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Moving Average Reward (window=50)
    ax2 = axes[0, 1]
    window_size = 50
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    ax2.plot(range(window_size-1, len(rewards)), moving_avg, linewidth=2, color='#A23B72')
    ax2.axvline(x=change_point, color='red', linestyle='--', linewidth=2, label='Environment Change')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title(f'Moving Average Reward (window={window_size})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Action Selection Frequency - Before Change
    ax3 = axes[1, 0]
    actions_before = np.array(actions_before_change)
    action_counts_before = [np.sum(actions_before == i) for i in range(3)]
    action_percentages_before = [count / len(actions_before) * 100 for count in action_counts_before]
    
    colors = ['#F18F01', '#C73E1D', '#6A994E']
    bars_before = ax3.bar(['Action 0', 'Action 1', 'Action 2'], action_percentages_before, 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax3.set_title(f'Action Selection Frequency - Before Change\n(Steps 0-{change_point-1})', 
                  fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars_before, action_percentages_before):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Action Selection Frequency - After Change
    ax4 = axes[1, 1]
    actions_after = np.array(actions_after_change)
    action_counts_after = [np.sum(actions_after == i) for i in range(3)]
    action_percentages_after = [count / len(actions_after) * 100 for count in action_counts_after]
    
    bars_after = ax4.bar(['Action 0', 'Action 1', 'Action 2'], action_percentages_after, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax4.set_title(f'Action Selection Frequency - After Change\n(Steps {change_point}-{len(actions_taken)-1})', 
                  fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars_after, action_percentages_after):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('q_learning_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'q_learning_results.png'")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("ACTION SELECTION SUMMARY")
    print("=" * 60)
    print("\nBefore Environment Change:")
    for i in range(3):
        print(f"  Action {i}: {action_counts_before[i]:4d} times ({action_percentages_before[i]:5.1f}%)")
    
    print("\nAfter Environment Change:")
    for i in range(3):
        print(f"  Action {i}: {action_counts_after[i]:4d} times ({action_percentages_after[i]:5.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    # Run experiment with 1000 steps
    results = run_experiment(
        n_steps=1000,
        learning_rate=0.1,
        epsilon=0.1,
        seed=42
    )
    
    # Plot results
    plot_results(results)
