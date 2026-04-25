"""
Network architecture reference for the SAC policy.

SB3's SAC(policy="MlpPolicy", policy_kwargs=dict(net_arch=[512, 512])) builds:
  - Actor:  obs_dim → 512 → 512 → 2*action_dim  (mean + log_std, tanh-squashed)
  - Critic: (obs_dim + action_dim) → 512 → 512 → 1  (x2 for twin Q)

The flat 41-dim observation fed to the policy is:
  [lidar_bins(36), dist_norm, cos_goal, sin_goal, prev_lv, prev_av]

Goal-relative features (dist, cos_goal, sin_goal) are already in robot frame
as computed by env_bridge_node, so no additional coordinate rotation is needed.
"""
