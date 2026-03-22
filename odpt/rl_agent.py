# rl_agent.py
# PPO agent with action masking for the DARP dispatch problem.
#
# Architecture:
#   Shared feature extractor (2 hidden layers) →
#     Actor head (masked softmax → action distribution)
#     Critic head (scalar value estimate)
#
# The action mask is applied BEFORE the softmax so the agent can
# never select an infeasible action.  This is the standard approach
# for constrained combinatorial RL (Huang et al., 2020).
#
# Dependencies: torch, numpy, gymnasium
#
# Usage:
#   agent = PPOAgent(obs_dim=OBS_SIZE, act_dim=MAX_ACTIONS)
#   action = agent.select_action(obs, action_mask)
#   ...collect trajectory...
#   agent.update(batch)

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. RL agent requires torch.")


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        """
        Shared-backbone actor-critic network for masked discrete actions.

        Architecture:
          obs → Linear(256) → ReLU → Linear(256) → ReLU →
            ├─ Actor:  Linear(act_dim) → masked softmax → π(a|s)
            └─ Critic: Linear(1) → V(s)

        The actor output is masked before softmax: infeasible actions
        get logit = -inf, so they receive probability zero.
        """

        def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.actor = nn.Linear(hidden, act_dim)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, obs: torch.Tensor, mask: torch.Tensor):
            """
            Parameters
            ----------
            obs  : (batch, obs_dim)
            mask : (batch, act_dim) — 1 = feasible, 0 = infeasible

            Returns
            -------
            logits : (batch, act_dim) — masked (infeasible = -inf)
            value  : (batch, 1)
            """
            features = self.shared(obs)
            logits = self.actor(features)

            # Apply action mask: set infeasible logits to -inf
            # so softmax assigns them probability zero.
            logits = logits.masked_fill(mask == 0, float("-inf"))

            value = self.critic(features)
            return logits, value


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one epoch of experience for PPO updates.

    Each entry corresponds to one dispatch decision (one step of the env).
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []       # action masks
        self.dones = []       # episode termination flags

    def store(self, obs, action, log_prob, reward, value, mask, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimization with clipped surrogate objective
    and action masking.

    Hyperparameters follow standard PPO recommendations with
    adjustments for the DARP domain:
      - Smaller batch sizes (episodes have ~200-400 steps)
      - Higher entropy coefficient (encourage exploration of
        diverse insertion positions)
      - Moderate clipping (ε=0.2, standard)

    Parameters
    ----------
    obs_dim : int
        Observation vector size.
    act_dim : int
        Maximum number of discrete actions (including reject).
    lr : float
        Learning rate for Adam optimiser.
    gamma : float
        Discount factor.  Set close to 1.0 because each step's
        reward is meaningful and episodes are finite.
    gae_lambda : float
        GAE λ for advantage estimation.
    clip_eps : float
        PPO clipping parameter.
    entropy_coef : float
        Entropy bonus coefficient (encourages exploration).
    value_coef : float
        Value loss coefficient.
    max_grad_norm : float
        Gradient clipping norm.
    n_epochs : int
        Number of passes over the rollout buffer per update.
    batch_size : int
        Mini-batch size for PPO updates.
    hidden : int
        Hidden layer size.
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden: int = 256,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PPOAgent")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.network = ActorCritic(obs_dim, act_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """
        Select an action given observation and mask.

        Returns
        -------
        action    : int
        log_prob  : float
        value     : float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        logits, value = self.network(obs_t, mask_t)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return (
            action.item(),
            log_prob.item(),
            value.squeeze().item(),
        )

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self, rewards, values, dones):
        """
        Generalised Advantage Estimation (GAE-λ).

        Returns advantages and returns (advantage + value).
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0  # terminal
            else:
                next_value = values[t + 1]

            non_terminal = 1.0 - float(dones[t])
            delta = (
                rewards[t]
                + self.gamma * next_value * non_terminal
                - values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        Run PPO update on the collected rollout buffer.

        Returns a dict of training metrics (losses, entropy, etc.)
        for logging.
        """
        buf = self.buffer
        n = len(buf)
        if n == 0:
            return {}

        # Compute GAE
        advantages, returns = self._compute_gae(
            buf.rewards, buf.values, buf.dones
        )

        # Convert to tensors
        obs_t = torch.FloatTensor(np.array(buf.obs)).to(self.device)
        act_t = torch.LongTensor(buf.actions).to(self.device)
        old_log_t = torch.FloatTensor(buf.log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        mask_t = torch.FloatTensor(np.array(buf.masks)).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            # Shuffle and create mini-batches
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_act = act_t[idx]
                b_old_log = old_log_t[idx]
                b_adv = adv_t[idx]
                b_ret = ret_t[idx]
                b_mask = mask_t[idx]

                # Forward pass
                logits, values = self.network(b_obs, b_mask)
                dist = Categorical(logits=logits)
                new_log_prob = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_prob - b_old_log)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), b_ret)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss":  total_value_loss / max(n_updates, 1),
            "entropy":     total_entropy / max(n_updates, 1),
            "buffer_size": n,
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
