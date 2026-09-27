import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.sac_actor import SACActor
from models.sac_critic import SACCritic


class SACAgent:

    def __init__(
        self,
        state_dim=256,
        action_dim=2,
        hidden_dim=256,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device="cpu"
    ):

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.grad_clip_norm = 5.0

        self.device = torch.device(device)

        # --------------------------------------------------
        # Automatic entropy temperature (alpha)
        #
        # A fixed alpha is brittle: too small -> policy
        # collapses to near-deterministic too early (poor
        # exploration); too large -> policy stays too random
        # to ever exploit useful actions. Standard SAC
        # (Haarnoja et al. 2018, "Soft Actor-Critic
        # Algorithms and Applications") instead learns
        # log_alpha so that the policy's entropy tracks a
        # target, usually -action_dim.
        # --------------------------------------------------

        self.target_entropy = -float(action_dim)

        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=actor_lr
        )

        # --------------------------------------------------
        # Actor
        # --------------------------------------------------

        self.actor = SACActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # --------------------------------------------------
        # Two critics
        # --------------------------------------------------

        self.critic_1 = SACCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.critic_2 = SACCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # --------------------------------------------------
        # Target critics
        # --------------------------------------------------

        self.target_critic_1 = copy.deepcopy(
            self.critic_1
        )

        self.target_critic_2 = copy.deepcopy(
            self.critic_2
        )

        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)

        # --------------------------------------------------
        # Optimizers
        # --------------------------------------------------

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(),
            lr=critic_lr
        )

        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(),
            lr=critic_lr
        )

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def compute_target_q(
        self,
        rewards,
        next_states,
        dones
    ):

        # --------------------------------------------------
        # 1. Sample next action from Actor
        # --------------------------------------------------

        next_actions, next_log_prob, _, _ = (
            self.actor.sample(next_states)
        )

        # --------------------------------------------------
        # 2. Compute target Q-values
        # --------------------------------------------------

        target_q1 = self.target_critic_1(
            next_states,
            next_actions
        )

        target_q2 = self.target_critic_2(
            next_states,
            next_actions
        )

        # --------------------------------------------------
        # 3. Take conservative estimate
        # --------------------------------------------------

        min_target_q = torch.min(
            target_q1,
            target_q2
        )

        # --------------------------------------------------
        # 4. Entropy-regularized target
        # --------------------------------------------------

        target_value = (
            min_target_q
            - self.alpha * next_log_prob
        )

        # --------------------------------------------------
        # 5. Bellman target
        # --------------------------------------------------

        target_q = (
            rewards
            + self.gamma
            * (1.0 - dones)
            * target_value
        )

        return target_q    
    def update_critics(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones
    ):

        # --------------------------------------------------
        # 1. Compute target
        # --------------------------------------------------

        target_q = self.compute_target_q(
            rewards,
            next_states,
            dones
        )

        # --------------------------------------------------
        # 2. Current Q-values
        # --------------------------------------------------

        current_q1 = self.critic_1(
            states,
            actions
        )

        current_q2 = self.critic_2(
            states,
            actions
        )

        # --------------------------------------------------
        # 3. Critic losses
        # --------------------------------------------------

        critic_1_loss = F.mse_loss(
            current_q1,
            target_q
        )

        critic_2_loss = F.mse_loss(
            current_q2,
            target_q
        )

        # --------------------------------------------------
        # 4. Update Critic 1
        # --------------------------------------------------

        self.critic_1_optimizer.zero_grad()

        critic_1_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic_1.parameters(),
            self.grad_clip_norm
        )

        self.critic_1_optimizer.step()

        # --------------------------------------------------
        # 5. Update Critic 2
        # --------------------------------------------------

        self.critic_2_optimizer.zero_grad()

        critic_2_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic_2.parameters(),
            self.grad_clip_norm
        )

        self.critic_2_optimizer.step()

        return (
            critic_1_loss.item(),
            critic_2_loss.item()
        )

    def update_actor(self, states):
        # Freeze critics during actor update
        for parameter in self.critic_1.parameters():
            parameter.requires_grad = False

        for parameter in self.critic_2.parameters():
            parameter.requires_grad = False

        # Sample actions from the current policy
        actions, log_prob, _, _ = self.actor.sample(states)

        # Evaluate sampled actions
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        # SAC uses the smaller Q-value
        min_q = torch.min(q1, q2)

        # Actor loss (use current alpha, detached — the
        # temperature is optimized separately below)
        actor_loss = (
            self.alpha * log_prob - min_q
        ).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.grad_clip_norm
        )

        self.actor_optimizer.step()

        # Unfreeze critics
        for parameter in self.critic_1.parameters():
            parameter.requires_grad = True

        for parameter in self.critic_2.parameters():
            parameter.requires_grad = True

        # --------------------------------------------------
        # Update entropy temperature (alpha)
        #
        # log_prob here is detached from the actor graph
        # w.r.t. this loss's purpose: we only want gradients
        # flowing into log_alpha, not into the actor/critics.
        # --------------------------------------------------

        alpha_loss = -(
            self.log_alpha
            * (log_prob.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()

    @torch.no_grad()
    def soft_update(self, source, target):
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters()
        ):
            target_param.copy_(
                self.tau * source_param
                + (1.0 - self.tau) * target_param
            )

    def update(self, states, actions, rewards, next_states, dones):

        # 1. Update both critics
        critic_1_loss, critic_2_loss = self.update_critics(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

        # 2. Update actor (and entropy temperature)
        actor_loss, alpha_loss = self.update_actor(states)

        # 3. Soft update target critics
        self.soft_update(
            self.critic_1,
            self.target_critic_1
        )

        self.soft_update(
            self.critic_2,
            self.target_critic_2
        )

        return {
            "critic_1_loss": critic_1_loss,
            "critic_2_loss": critic_2_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha.item()
        }
