import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.environment import NonStationaryBandit
from src.agent import QLearningAgent


def run_experiment(epsilon, n_steps=1000, learning_rate=0.1, seed=42):
    # Create environment and agent
    env = NonStationaryBandit(n_steps=n_steps, seed=seed)
    agent = QLearningAgent(n_actions=3, learning_rate=learning_rate, epsilon=epsilon, seed=seed)
    
    # Storage for results
    rewards = []
    cumulative_rewards = []
    actions_taken = []
    
    # Track actions before and after environment change
    change_point = env.change_point
    actions_before_change = []
    actions_after_change = []
    
    print(f"\nRunning experiment with ε = {epsilon}")
    print("-" * 60)
    
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
        
        # Track actions before/after change
        if step < change_point:
            actions_before_change.append(action)
        else:
            actions_after_change.append(action)
        
        if done:
            break
    
    print(f"Total reward: {total_reward}")
    print(f"Average reward: {total_reward / n_steps:.3f}")
    print(f"Final Q-values: {agent.q_values.round(3)}")
    
    # Return results
    return {
        'epsilon': epsilon,
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
        'actions_taken': actions_taken,
        'actions_before_change': actions_before_change,
        'actions_after_change': actions_after_change,
        'change_point': change_point,
        'total_reward': total_reward,
        'final_q_values': agent.q_values.copy(),
        'action_counts': agent.action_counts.copy()
    }


def plot_comparison(results_low, results_high):
    """
    Plot comparison of experiments with different epsilon values.
    
    Args:
        results_low (dict): Results from low epsilon experiment
        results_high (dict): Results from high epsilon experiment
    """
    epsilon_low = results_low['epsilon']
    epsilon_high = results_high['epsilon']
    change_point = results_low['change_point']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Q-Learning Comparison: ε = {epsilon_low} vs ε = {epsilon_high}', 
                 fontsize=16, fontweight='bold')
    
    # Colors for the two experiments
    color_low = '#2E86AB'
    color_high = '#A23B72'
    
    # Plot 1: Cumulative Reward Comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results_low['cumulative_rewards'], linewidth=2, color=color_low, 
             label=f'ε = {epsilon_low} (less exploration)', alpha=0.8)
    ax1.plot(results_high['cumulative_rewards'], linewidth=2, color=color_high, 
             label=f'ε = {epsilon_high} (more exploration)', alpha=0.8)
    ax1.axvline(x=change_point, color='red', linestyle='--', linewidth=2, 
                label='Environment Change')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Cumulative Reward', fontsize=12)
    ax1.set_title('Cumulative Reward Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper left')
    
    # Plot 2: Moving Average Reward Comparison
    ax2 = fig.add_subplot(gs[1, :])
    window_size = 50
    moving_avg_low = np.convolve(results_low['rewards'], np.ones(window_size)/window_size, mode='valid')
    moving_avg_high = np.convolve(results_high['rewards'], np.ones(window_size)/window_size, mode='valid')
    
    ax2.plot(range(window_size-1, len(results_low['rewards'])), moving_avg_low, 
             linewidth=2, color=color_low, label=f'ε = {epsilon_low}', alpha=0.8)
    ax2.plot(range(window_size-1, len(results_high['rewards'])), moving_avg_high, 
             linewidth=2, color=color_high, label=f'ε = {epsilon_high}', alpha=0.8)
    ax2.axvline(x=change_point, color='red', linestyle='--', linewidth=2, 
                label='Environment Change')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title(f'Moving Average Reward (window={window_size})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Action Selection - Low Epsilon (Before Change)
    ax3 = fig.add_subplot(gs[2, 0])
    actions_before_low = np.array(results_low['actions_before_change'])
    action_counts_before_low = [np.sum(actions_before_low == i) for i in range(3)]
    action_pct_before_low = [count / len(actions_before_low) * 100 for count in action_counts_before_low]
    
    colors = ['#F18F01', '#C73E1D', '#6A994E']
    bars = ax3.bar(['Action 0', 'Action 1', 'Action 2'], action_pct_before_low, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Selection Frequency (%)', fontsize=11)
    ax3.set_title(f'ε = {epsilon_low} - Before Change\n(Steps 0-{change_point-1})', 
                  fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, action_pct_before_low):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Action Selection - High Epsilon (Before Change)
    ax4 = fig.add_subplot(gs[2, 1])
    actions_before_high = np.array(results_high['actions_before_change'])
    action_counts_before_high = [np.sum(actions_before_high == i) for i in range(3)]
    action_pct_before_high = [count / len(actions_before_high) * 100 for count in action_counts_before_high]
    
    bars = ax4.bar(['Action 0', 'Action 1', 'Action 2'], action_pct_before_high, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Selection Frequency (%)', fontsize=11)
    ax4.set_title(f'ε = {epsilon_high} - Before Change\n(Steps 0-{change_point-1})', 
                  fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, action_pct_before_high):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Save figure
    plt.savefig('experiments/epsilon_comparison.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Comparison plot saved as 'experiments/epsilon_comparison.png'")
    print("=" * 60)
    
    # Create second figure for after-change comparison
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f'Action Selection After Environment Change', fontsize=14, fontweight='bold')
    
    # Plot 5: Action Selection - Low Epsilon (After Change)
    actions_after_low = np.array(results_low['actions_after_change'])
    action_counts_after_low = [np.sum(actions_after_low == i) for i in range(3)]
    action_pct_after_low = [count / len(actions_after_low) * 100 for count in action_counts_after_low]
    
    bars = ax5.bar(['Action 0', 'Action 1', 'Action 2'], action_pct_after_low, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Selection Frequency (%)', fontsize=11)
    ax5.set_title(f'ε = {epsilon_low} - After Change\n(Steps {change_point}-999)', 
                  fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 100])
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, action_pct_after_low):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 6: Action Selection - High Epsilon (After Change)
    actions_after_high = np.array(results_high['actions_after_change'])
    action_counts_after_high = [np.sum(actions_after_high == i) for i in range(3)]
    action_pct_after_high = [count / len(actions_after_high) * 100 for count in action_counts_after_high]
    
    bars = ax6.bar(['Action 0', 'Action 1', 'Action 2'], action_pct_after_high, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Selection Frequency (%)', fontsize=11)
    ax6.set_title(f'ε = {epsilon_high} - After Change\n(Steps {change_point}-999)', 
                  fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 100])
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, action_pct_after_high):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiments/epsilon_comparison_after.png', dpi=300, bbox_inches='tight')
    print("After-change plot saved as 'experiments/epsilon_comparison_after.png'")
    print("=" * 60)


def print_summary(results_low, results_high):
    """
    Print detailed summary comparing both experiments.
    
    Args:
        results_low (dict): Results from low epsilon experiment
        results_high (dict): Results from high epsilon experiment
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<40} {'ε = ' + str(results_low['epsilon']):<15} {'ε = ' + str(results_high['epsilon']):<15}")
    print("-" * 60)
    print(f"{'Total Reward':<40} {results_low['total_reward']:<15} {results_high['total_reward']:<15}")
    print(f"{'Average Reward':<40} {results_low['total_reward']/1000:<15.3f} {results_high['total_reward']/1000:<15.3f}")
    
    print("\n" + "-" * 60)
    print("Action Selection Before Change (Steps 0-499):")
    print("-" * 60)
    
    actions_before_low = np.array(results_low['actions_before_change'])
    actions_before_high = np.array(results_high['actions_before_change'])
    
    for i in range(3):
        count_low = np.sum(actions_before_low == i)
        count_high = np.sum(actions_before_high == i)
        pct_low = count_low / len(actions_before_low) * 100
        pct_high = count_high / len(actions_before_high) * 100
        print(f"  Action {i}: {count_low:4d} ({pct_low:5.1f}%)        {count_high:4d} ({pct_high:5.1f}%)")
    
    print("\n" + "-" * 60)
    print("Action Selection After Change (Steps 500-999):")
    print("-" * 60)
    
    actions_after_low = np.array(results_low['actions_after_change'])
    actions_after_high = np.array(results_high['actions_after_change'])
    
    for i in range(3):
        count_low = np.sum(actions_after_low == i)
        count_high = np.sum(actions_after_high == i)
        pct_low = count_low / len(actions_after_low) * 100
        pct_high = count_high / len(actions_after_high) * 100
        print(f"  Action {i}: {count_low:4d} ({pct_low:5.1f}%)        {count_high:4d} ({pct_high:5.1f}%)")
    
    print("\n" + "-" * 60)
    print("Final Q-values:")
    print("-" * 60)
    print(f"  ε = {results_low['epsilon']}: {results_low['final_q_values'].round(3)}")
    print(f"  ε = {results_high['epsilon']}: {results_high['final_q_values'].round(3)}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("EPSILON COMPARISON EXPERIMENT")
    print("=" * 60)
    print("\nComparing Q-learning performance with different exploration rates:")
    print("  - Low exploration: ε = 0.1 (10% random actions)")
    print("  - High exploration: ε = 0.3 (30% random actions)")
    print("\nEnvironment Details:")
    print("  - Before change (steps 0-499): [0.3, 0.6, 0.4] → Action 1 optimal")
    print("  - After change (steps 500-999): [0.7, 0.3, 0.5] → Action 0 optimal")
    print("=" * 60)
    
    # Run experiments
    results_low = run_experiment(epsilon=0.1, n_steps=1000, learning_rate=0.1, seed=42)
    results_high = run_experiment(epsilon=0.3, n_steps=1000, learning_rate=0.1, seed=42)
    
    # Plot comparison
    plot_comparison(results_low, results_high)
    
    # Print summary
    print_summary(results_low, results_high)
