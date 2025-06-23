from __future__ import absolute_import, division, print_function

import argparse
import os
from collections import namedtuple
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .rl_kg_env import BatchKGEnvironment
from .rl_utils import TMP_DIR, USERID, get_logger, set_random_seed

logger = None

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for reinforcement learning.
    
    Combines policy (actor) and value function (critic) estimation
    in a single network with shared feature layers.
    """
    
    def __init__(self, state_dim: int, act_dim: int, gamma: float = 0.99, hidden_sizes: List[int] = [512, 256]):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            act_dim: Dimension of action space
            gamma: Discount factor for future rewards
            hidden_sizes: List of hidden layer sizes
        """
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        # Network layers
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        # Training buffers
        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            inputs: Tuple of (state, action_mask) tensors
            
        Returns:
            Tuple of (action_probabilities, state_values)
        """
        state, act_mask = inputs
        
        # Shared feature extraction
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)

        # Actor head (policy)
        actor_logits = self.actor(x)
        actor_logits[~act_mask] = -999999.0  # Mask invalid actions
        act_probs = F.softmax(actor_logits, dim=-1)

        # Critic head (value function)
        state_values = self.critic(x)
        
        return act_probs, state_values

    def select_action(self, batch_state: np.ndarray, batch_act_mask: np.ndarray, device: torch.device) -> List[int]:
        """
        Select actions for a batch of states using current policy.
        
        Args:
            batch_state: Batch of state vectors
            batch_act_mask: Batch of action masks
            device: PyTorch device for computation
            
        Returns:
            List of selected action indices
        """
        # Convert to tensors
        state = torch.FloatTensor(batch_state).to(device)
        act_mask = torch.BoolTensor(batch_act_mask).to(device)

        # Get action probabilities and state values
        probs, value = self((state, act_mask))
        
        # Sample actions from categorical distribution
        m = Categorical(probs)
        acts = m.sample()
        
        # Ensure selected actions are valid
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1).bool()
        acts[~valid_idx] = 0  # Default to action 0 for invalid selections

        # Store for training
        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        
        return acts.cpu().numpy().tolist()

    def update(self, optimizer: torch.optim.Optimizer, device: torch.device, ent_weight: float) -> Tuple[float, float, float, float]:
        """
        Update the network using collected experiences.
        
        Args:
            optimizer: PyTorch optimizer
            device: PyTorch device for computation
            ent_weight: Weight for entropy regularization
            
        Returns:
            Tuple of (total_loss, actor_loss, critic_loss, entropy_loss)
        """
        # Check if we have any experiences
        if len(self.rewards) <= 0:
            self._clear_buffers()
            return 0.0, 0.0, 0.0, 0.0

        # Calculate discounted rewards
        batch_rewards = self._calculate_discounted_rewards(device)
        
        # Calculate losses
        actor_loss, critic_loss, entropy_loss = self._calculate_losses(batch_rewards)
        
        # Combine losses
        total_loss = actor_loss + critic_loss + ent_weight * entropy_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Clear buffers
        self._clear_buffers()

        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def _calculate_discounted_rewards(self, device: torch.device) -> torch.Tensor:
        """Calculate discounted cumulative rewards."""
        batch_rewards = np.vstack(self.rewards).T
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        
        # Apply discount factor
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]
        
        return batch_rewards

    def _calculate_losses(self, batch_rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate actor, critic, and entropy losses."""
        num_steps = batch_rewards.shape[1]
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
        for i in range(num_steps):
            log_prob, value = self.saved_actions[i]
            advantage = batch_rewards[:, i] - value.squeeze(1)
            
            # Actor loss (policy gradient)
            actor_loss += -log_prob * advantage.detach()
            
            # Critic loss (value function)
            critic_loss += advantage.pow(2)
            
            # Entropy loss (exploration)
            entropy_loss += -self.entropy[i]
        
        # Average over batch
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        
        return actor_loss, critic_loss, entropy_loss

    def _clear_buffers(self) -> None:
        """Clear training buffers."""
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]


class ACDataLoader:
    """
    Data loader for Actor-Critic training that provides batches of user IDs.
    """
    
    def __init__(self, uids: List[int], batch_size: int):
        """
        Initialize data loader.
        
        Args:
            uids: List of user IDs
            batch_size: Size of each batch
        """
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self) -> None:
        """Reset the data loader for a new epoch."""
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self) -> bool:
        """Check if there are more batches available."""
        return self._has_next

    def get_batch(self) -> Optional[List[int]]:
        """
        Get the next batch of user IDs.
        
        Returns:
            List of user IDs or None if no more batches
        """
        if not self._has_next:
            return None
            
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        
        # Update state for next batch
        self._has_next = end_idx < self.num_users
        self._start_idx = end_idx
        
        return batch_uids.tolist()


def train(args: argparse.Namespace) -> None:
    """
    Main training function for the Actor-Critic agent.
    
    Args:
        args: Configuration arguments
    """
    # Initialize environment and data loader
    env = BatchKGEnvironment(
        args.dataset,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
    )
    
    uids = list(env.kg(USERID).keys())
    dataloader = ACDataLoader(uids, args.batch_size)
    
    # Initialize model and optimizer
    model = ActorCritic(
        env.state_dim, 
        env.act_dim, 
        gamma=args.gamma, 
        hidden_sizes=args.hidden
    ).to(args.device)
    
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training metrics
    training_metrics = _initialize_training_metrics()
    step = 0
    model.train()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Starting epoch {epoch}/{args.epochs}")
        
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            
            # Run episode for current batch
            _run_episode(env, model, batch_uids, args)
            
            # Update learning rate
            _update_learning_rate(optimizer, step, args, len(uids))
            
            # Update model
            _update_model_and_metrics(model, optimizer, training_metrics, args, step)
            step += 1

            # Report progress
            if step > 0 and step % 100 == 0:
                _report_training_progress(training_metrics, epoch, step, args.batch_size)

        # Save model checkpoint
        _save_model_checkpoint(model, epoch, args)


def _initialize_training_metrics() -> dict:
    """Initialize containers for tracking training metrics."""
    return {
        'total_losses': [],
        'total_plosses': [],
        'total_vlosses': [],
        'total_entropy': [],
        'total_rewards': []
    }


def _run_episode(env: BatchKGEnvironment, model: ActorCritic, batch_uids: List[int], args: argparse.Namespace) -> None:
    """Run a single episode for the current batch."""
    batch_state = env.reset(batch_uids)
    done = False
    
    while not done:
        batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)
        batch_act_idx = model.select_action(batch_state, batch_act_mask, args.device)
        batch_state, batch_reward, done = env.batch_step(batch_act_idx)
        model.rewards.append(batch_reward)


def _update_learning_rate(optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace, num_users: int) -> None:
    """Update learning rate with linear decay."""
    total_steps = args.epochs * num_users / args.batch_size
    lr = args.lr * max(1e-4, 1.0 - float(step) / total_steps)
    
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _update_model_and_metrics(
    model: ActorCritic, 
    optimizer: torch.optim.Optimizer, 
    metrics: dict, 
    args: argparse.Namespace,
    step: int
) -> None:
    """Update model parameters and collect training metrics."""
    # Store total reward before update
    metrics['total_rewards'].append(np.sum(model.rewards))
    
    # Update model
    loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight)
    
    # Store losses
    metrics['total_losses'].append(loss)
    metrics['total_plosses'].append(ploss)
    metrics['total_vlosses'].append(vloss)
    metrics['total_entropy'].append(eloss)


def _report_training_progress(metrics: dict, epoch: int, step: int, batch_size: int) -> None:
    """Report training progress and reset metrics."""
    avg_reward = np.mean(metrics['total_rewards']) / batch_size
    avg_loss = np.mean(metrics['total_losses'])
    avg_ploss = np.mean(metrics['total_plosses'])
    avg_vloss = np.mean(metrics['total_vlosses'])
    avg_entropy = np.mean(metrics['total_entropy'])
    
    # Reset metrics
    for key in metrics:
        metrics[key].clear()
    
    logger.info(
        f"epoch/step={epoch:d}/{step:d}"
        f" | loss={avg_loss:.5f}"
        f" | ploss={avg_ploss:.5f}"
        f" | vloss={avg_vloss:.5f}"
        f" | entropy={avg_entropy:.5f}"
        f" | reward={avg_reward:.5f}"
    )


def _save_model_checkpoint(model: ActorCritic, epoch: int, args: argparse.Namespace) -> None:
    """Save model checkpoint."""
    policy_file = f"{args.log_dir}/policy_model_epoch_{epoch}.ckpt"
    logger.info(f"Save model to {policy_file}")
    torch.save(model.state_dict(), policy_file)


def train_agent_rl(dataset: str, seed: int) -> None:
    """
    Main entry point for training the RL agent.
    
    Args:
        dataset: Dataset name
        seed: Random seed for reproducibility
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=dataset, help="Dataset name (set automatically).")
    parser.add_argument("--name", type=str, default="train_agent", help="directory name.")
    parser.add_argument("--seed", type=int, default=seed, help="random seed.")
    parser.add_argument("--gpu", type=str, default="0", help="gpu device.")
    parser.add_argument("--epochs", type=int, default=50, help="Max number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate.")
    parser.add_argument("--max_acts", type=int, default=250, help="Max number of actions.")
    parser.add_argument("--max_path_len", type=int, default=3, help="Max path length.")
    parser.add_argument("--gamma", type=float, default=0.99, help="reward discount factor.")
    parser.add_argument("--ent_weight", type=float, default=1e-3, help="weight factor for entropy loss")
    parser.add_argument("--act_dropout", type=float, default=0.5, help="action dropout rate.")
    parser.add_argument("--state_history", type=int, default=1, help="state history length")
    parser.add_argument("--hidden", type=int, nargs="*", default=[512, 256], help="hidden layer sizes")
    args = parser.parse_args()

    # Set up device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    # Set up logging directory
    args.log_dir = f"{TMP_DIR[args.dataset]}/{args.name}"
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # Set up logger
    global logger
    logger = get_logger(args.log_dir + "/train_log.txt")
    logger.info(args)

    # Set random seed and start training
    set_random_seed(args.seed)
    train(args)
