from __future__ import absolute_import, division, print_function

import random
from typing import List, Optional, Tuple

import numpy as np

from .rl_utils import ITEMID, KG_RELATION, PATH_PATTERN, RATED, SELF_LOOP, TITLE, USERID, WATCHED, load_embed, load_kg


class KGState:
    """
    Knowledge Graph State representation for RL agent.

    Generates state vectors by concatenating user embeddings with
    current node and historical information based on history length.
    """

    def __init__(self, embed_size: int, history_len: int = 1):
        """
        Initialize state generator.

        Args:
            embed_size: Dimension of embeddings
            history_len: Number of historical steps to include (0, 1, or 2)

        Raises:
            Exception: If history_len is not in {0, 1, 2}
        """
        self.embed_size = embed_size
        self.history_len = history_len

        # Calculate state dimension based on history length
        if history_len == 0:
            self.dim = 2 * embed_size  # user + current node
        elif history_len == 1:
            self.dim = 4 * embed_size  # user + current + last node + last relation
        elif history_len == 2:
            self.dim = 6 * embed_size  # user + current + last + older (node + relation)
        else:
            raise Exception("history length should be one of {0, 1, 2}")

    def __call__(
        self,
        user_embed: np.ndarray,
        node_embed: np.ndarray,
        last_node_embed: np.ndarray,
        last_relation_embed: np.ndarray,
        older_node_embed: np.ndarray,
        older_relation_embed: np.ndarray,
    ) -> np.ndarray:
        """
        Generate state vector from embeddings.

        Args:
            user_embed: User embedding vector
            node_embed: Current node embedding
            last_node_embed: Previous node embedding
            last_relation_embed: Previous relation embedding
            older_node_embed: Older node embedding (for history_len=2)
            older_relation_embed: Older relation embedding (for history_len=2)

        Returns:
            Concatenated state vector

        Raises:
            Exception: If history_len is invalid
        """
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed])
        else:
            raise Exception("Invalid history length in state generation")


class BatchKGEnvironment:
    """
    Batch Knowledge Graph Environment for RL training.

    Manages multiple user sessions simultaneously, providing state representations,
    available actions, and rewards based on knowledge graph traversal patterns.
    """

    def __init__(self, dataset_str: str, max_acts: int, max_path_len: int = 3, state_history: int = 1):
        """
        Initialize the KG environment.

        Args:
            dataset_str: Dataset name for loading KG and embeddings
            max_acts: Maximum number of actions per step
            max_path_len: Maximum path length before episode termination
            state_history: History length for state representation
        """
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # +1 for self-loop action
        self.max_num_nodes = max_path_len + 1

        # Load knowledge graph and embeddings
        self.kg = load_kg(dataset_str)
        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USERID].shape[1]

        # Add self-loop embedding (zero vector with zero bias)
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)

        # Initialize state generator
        self.state_gen = KGState(self.embed_size, history_len=state_history)
        self.state_dim = self.state_gen.dim

        # Compute user-product scores for reward scaling
        self._compute_user_product_scales()

        # Initialize valid path patterns
        self._initialize_path_patterns()

        # Episode state variables
        self._batch_path = None
        self._batch_curr_actions = None
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False

    def _compute_user_product_scales(self) -> None:
        """Compute user-product interaction scales for reward normalization."""
        u_p_scores = np.dot(self.embeds[USERID] + self.embeds[WATCHED][0], self.embeds[ITEMID].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

    def _initialize_path_patterns(self) -> None:
        """Initialize valid reasoning path patterns."""
        self.patterns = []
        for pattern_id in [1, 2, 11, 12, 13, 14, 15]:
            pattern = PATH_PATTERN[pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]
            if pattern_id == 1:
                pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

    def _has_pattern(self, path: List[Tuple]) -> bool:
        """Check if a path matches any valid reasoning pattern."""
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path: List[List[Tuple]]) -> List[bool]:
        """Check patterns for a batch of paths."""
        return [self._has_pattern(path) for path in batch_path]

    def _get_actions(self, path: List[Tuple], done: bool) -> List[Tuple]:
        """
        Compute available actions for current node in path.

        Args:
            path: Current path as list of (relation, node_type, node_id) tuples
            done: Whether episode is finished

        Returns:
            List of (relation, node_id) action tuples
        """
        _, curr_node_type, curr_node_id = path[-1]
        actions = [(SELF_LOOP, curr_node_id)]

        # If episode is done, only self-loop action is available
        if done:
            return actions

        # Get candidate actions from knowledge graph
        candidate_acts = self._get_candidate_actions(path, curr_node_type, curr_node_id)

        # If no candidates, return only self-loop
        if not candidate_acts:
            return actions

        # If candidates fit within max_acts, return all
        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        # Trim actions using embedding-based scoring
        candidate_acts = self._trim_actions_by_score(path, candidate_acts)
        actions.extend(candidate_acts)
        return actions

    def _get_candidate_actions(self, path: List[Tuple], curr_node_type: str, curr_node_id: int) -> List[Tuple]:
        """Get candidate actions excluding visited nodes."""
        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []
        visited_nodes = set([(v[1], v[2]) for v in path])

        for r in relations_nodes:
            next_node_type = KG_RELATION[curr_node_type][r]
            next_node_ids = relations_nodes[r]
            next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        return candidate_acts

    def _trim_actions_by_score(self, path: List[Tuple], candidate_acts: List[Tuple]) -> List[Tuple]:
        """Trim candidate actions using embedding-based scoring."""
        user_embed = self.embeds[USERID][path[0][-1]]
        scores = []

        for r, next_node_id in candidate_acts:
            score = self._compute_action_score(path, user_embed, r, next_node_id)
            scores.append(score)

        # Select top-k actions by score
        candidate_idxs = np.argsort(scores)[-self.max_acts :]
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        return candidate_acts

    def _compute_action_score(self, path: List[Tuple], user_embed: np.ndarray, relation: str, next_node_id: int) -> float:
        """Compute score for a candidate action."""
        curr_node_type = path[-1][1]
        next_node_type = KG_RELATION[curr_node_type][relation]

        # Choose source embedding based on next node type
        if next_node_type == USERID:
            src_embed = user_embed
        elif next_node_type == ITEMID:
            src_embed = user_embed + self.embeds[WATCHED][0]
        elif next_node_type == TITLE:
            src_embed = user_embed + self.embeds[RATED][0]
        else:
            src_embed = user_embed + self.embeds[WATCHED][0] + self.embeds[relation][0]

        score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
        return score

    def _batch_get_actions(self, batch_path: List[List[Tuple]], done: bool) -> List[List[Tuple]]:
        """Get actions for a batch of paths."""
        return [self._get_actions(path, done) for path in batch_path]

    def _get_state(self, path: List[Tuple]) -> np.ndarray:
        """
        Generate state representation for a path.

        Args:
            path: Current path as list of (relation, node_type, node_id) tuples

        Returns:
            State vector combining user, current node, and history embeddings
        """
        user_embed = self.embeds[USERID][path[0][-1]]
        zero_embed = np.zeros(self.embed_size)

        if len(path) == 1:
            # Only user node in path
            return self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)

        # Extract path components
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_relation_embed, _ = self.embeds[last_relation]

        if len(path) == 2:
            # Only current and previous step
            older_relation, last_node_type, last_node_id = path[-2]
            last_node_embed = self.embeds[last_node_type][last_node_id]
            return self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed, zero_embed)

        # Full history available
        older_relation, last_node_type, last_node_id = path[-2]
        _, older_node_type, older_node_id = path[-3]

        last_node_embed = self.embeds[last_node_type][last_node_id]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]

        return self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed)

    def _batch_get_state(self, batch_path: List[List[Tuple]]) -> np.ndarray:
        """Generate state representations for a batch of paths."""
        batch_state = [self._get_state(path) for path in batch_path]
        return np.vstack(batch_state)

    def _get_reward(self, path: List[Tuple]) -> float:
        """
        Calculate reward for a path.

        Args:
            path: Current path as list of (relation, node_type, node_id) tuples

        Returns:
            Reward value (0.0 if path too short or invalid pattern,
            scaled score if ends at item)
        """
        # Require minimum path length
        if len(path) <= 2:
            return 0.0

        # Check if path follows valid reasoning pattern
        if not self._has_pattern(path):
            return 0.0

        # Calculate reward based on final node
        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]

        if curr_node_type == ITEMID:
            uid = path[0][-1]
            u_vec = self.embeds[USERID][uid] + self.embeds[WATCHED][0]
            p_vec = self.embeds[ITEMID][curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
            target_score = max(score, 0.0)

        return target_score

    def _batch_get_reward(self, batch_path: List[List[Tuple]]) -> np.ndarray:
        """Calculate rewards for a batch of paths."""
        batch_reward = [self._get_reward(path) for path in batch_path]
        return np.array(batch_reward)

    def _is_done(self) -> bool:
        """Check if episode should terminate (max path length reached)."""
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset(self, uids: Optional[List[int]] = None) -> np.ndarray:
        """
        Reset environment with new user IDs.

        Args:
            uids: List of user IDs to start episodes with.
                  If None, randomly selects one user.

        Returns:
            Initial state representations for the batch
        """
        if uids is None:
            all_uids = list(self.kg(USERID).keys())
            uids = [random.choice(all_uids)]

        # Initialize paths starting from users
        self._batch_path = [[(SELF_LOOP, USERID, uid)] for uid in uids]
        self._done = False

        # Compute initial state, actions, and rewards
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx: List[int]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Execute actions for the batch and update environment state.

        Args:
            batch_act_idx: List of action indices for each path in the batch

        Returns:
            Tuple of (next_states, rewards, done_flag)
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute actions for each path in batch
        for i, act_idx in enumerate(batch_act_idx):
            self._execute_action(i, act_idx)

        # Update environment state
        self._done = self._is_done()
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def _execute_action(self, batch_idx: int, act_idx: int) -> None:
        """Execute a single action for one path in the batch."""
        _, curr_node_type, curr_node_id = self._batch_path[batch_idx][-1]
        relation, next_node_id = self._batch_curr_actions[batch_idx][act_idx]

        if relation == SELF_LOOP:
            next_node_type = curr_node_type
        else:
            next_node_type = KG_RELATION[curr_node_type][relation]

        self._batch_path[batch_idx].append((relation, next_node_type, next_node_id))

    def batch_action_mask(self, dropout: float = 0.0) -> np.ndarray:
        """
        Generate action masks for the batch.

        Args:
            dropout: Probability of dropping non-essential actions (for exploration)

        Returns:
            Boolean mask array of shape [batch_size, act_dim]
        """
        batch_mask = []

        for actions in self._batch_curr_actions:
            act_mask = self._create_action_mask(actions, dropout)
            batch_mask.append(act_mask)

        return np.vstack(batch_mask)

    def _create_action_mask(self, actions: List[Tuple], dropout: float) -> np.ndarray:
        """Create action mask for a single set of actions."""
        act_idxs = list(range(len(actions)))

        # Apply dropout to non-essential actions (keep self-loop)
        if dropout > 0 and len(act_idxs) >= 5:
            keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
            tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
            act_idxs = [act_idxs[0]] + tmp

        # Create mask
        act_mask = np.zeros(self.act_dim, dtype=np.uint8)
        act_mask[act_idxs] = 1
        return act_mask

    def print_path(self) -> None:
        """Print current paths for debugging."""
        for path in self._batch_path:
            msg = f"Path: {path[0][1]}({path[0][2]})"
            for node in path[1:]:
                msg += f" =={node[0]}=> {node[1]}({node[2]})"
            print(msg)
