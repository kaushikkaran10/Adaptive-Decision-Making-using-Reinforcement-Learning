# Adaptive Decision Making with Reinforcement Learning

## Problem
How is a learning system able to change its decisions when the environment changes and cannot rely on past experience?

## Motivation
Real world systems from biological learning to recommendation engines arenon-stationary . Investigating how agents adapt to these changes is at the core of reinforcement learning and decision, making research.

## Method
We considered the problem as a non, stationary multi, armed bandit and implemented a Q, learning agent with epsilon, greedy exploration. The environment changes reward distributions in the middle of training to test adaptive behavior.

## Experiments
- Learning behavior before and after environment change
- Action selection frequency analysis
- Exploration rate comparison (ε = 0.1 vs ε = 0.3)

## Results
The agent learns an optimal policy before the environment change, experiences a drop in performance when reward distributions shift, and successfully adapts by modifying its action preferences. Higher exploration rates enable faster adaptation at the cost of short term stability.

## Conclusion
This project demonstrates how reinforcement learning agents adapt in non stationary environments through continuous learning and exploration.

## Limitations
The environment is simplified and does not capture complex real-world dynamics. Future work could extend this approach to contextual or multi-state environments.

