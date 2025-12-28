# Epsilon Comparison Experiment Results

### Exploration Rate Comparison (ε = 0.1 vs ε = 0.3)

We compared two reinforcement learning agents with different exploration rates to study how exploration influences adaptation in a non stationary environment.

### Observations

- Before the environment change, ε = 0.1 showed more stable performance due to greater exploitation of learned actions.
- After the environment change, ε = 0.3 adapted faster, recovering average reward more quickly.
- Higher exploration allowed the agent to detect changes sooner, at the cost of reduced short term stability.

### Interpretation

These results highlight the exploration exploitation trade off in reinforcement learning. While lower exploration improves stability in stationary environments, higher exploration is critical for adapting to non stationary conditions.
