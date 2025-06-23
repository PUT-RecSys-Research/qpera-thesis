from __future__ import absolute_import, division, print_function

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_utils import KG_RELATION, get_entities, get_entity_tail


class KnowledgeEmbedding(nn.Module):
    """
    TransE-based Knowledge Graph Embedding model.
    
    Implements the TransE algorithm for learning entity and relation embeddings
    in a knowledge graph using negative sampling loss.
    """
    
    def __init__(self, processed_dataset: Dict[str, Any], idx_to_relation_name_map: Dict[int, str], args):
        """
        Initialize Knowledge Graph Embedding model.
        
        Args:
            processed_dataset: Dictionary containing entity maps and distributions
            idx_to_relation_name_map: Mapping from relation indices to names
            args: Configuration arguments
        """
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.idx_to_relation_name_map = idx_to_relation_name_map

        # Validate input
        if not isinstance(processed_dataset, dict):
            raise TypeError("KnowledgeEmbedding expects processed_dataset to be a dictionary.")

        entity_maps = processed_dataset.get("entity_maps", {})
        distributions = processed_dataset.get("distributions", {})

        # Initialize entity embeddings
        self._create_entity_embeddings(entity_maps)
        
        # Initialize relation embeddings and info
        self.relation_info = {}
        self._create_relation_embeddings(entity_maps, distributions)

    def _create_entity_embeddings(self, entity_maps: Dict[str, Any]) -> None:
        """Create embedding layers for all entities."""
        print("Creating entity embeddings...")
        self.entity_names = get_entities()
        
        for entity_name in self.entity_names:
            map_data = entity_maps.get(entity_name, {})
            vocab_size = map_data.get("vocab_size", 0)
            
            if vocab_size == 0:
                print(f"Warning: Vocab size is 0 for entity '{entity_name}'. Skipping embedding creation.")
                continue
                
            embed = self._entity_embedding(vocab_size)
            layer_name = entity_name.lower()
            setattr(self, layer_name, embed)
            print(f"  Created embedding layer '{layer_name}' for entity '{entity_name}' with size {vocab_size + 1}")

    def _create_relation_embeddings(self, entity_maps: Dict[str, Any], distributions: Dict[str, np.ndarray]) -> None:
        """Create embedding layers and info for all relations."""
        print("Creating relation embeddings...")
        
        for head_entity_type, relations_in_head in KG_RELATION.items():
            for relation_name_const in relations_in_head:
                if relation_name_const in self.relation_info:
                    continue

                # Get relation information
                relation_info = self._get_relation_info(
                    head_entity_type, relation_name_const, entity_maps, distributions
                )
                
                if relation_info is None:
                    continue

                # Store relation info and create embeddings
                self.relation_info[relation_name_const] = relation_info
                self._create_single_relation_embedding(relation_name_const, relation_info["tail_vocab_size"])

    def _get_relation_info(
        self,
        head_entity_type: str,
        relation_name_const: str,
        entity_maps: Dict[str, Any],
        distributions: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, Any]]:
        """Get information for a single relation."""
        try:
            tail_entity_type_const = get_entity_tail(head_entity_type, relation_name_const)
        except KeyError as e:
            print(f"Warning: Could not determine tail entity for relation '{relation_name_const}': {e}")
            return None

        tail_map_data = entity_maps.get(tail_entity_type_const, {})
        tail_vocab_size = tail_map_data.get("vocab_size", 0)
        distrib_np = distributions.get(relation_name_const, None)

        if tail_vocab_size == 0:
            print(f"Warning: Tail vocab size is 0 for relation '{relation_name_const}'. Skipping relation layer creation.")
            return None
            
        if distrib_np is None:
            print(f"Warning: Distribution not found for relation '{relation_name_const}'. Skipping relation layer creation.")
            return None

        return {
            "head_entity": head_entity_type.lower(),
            "tail_entity": tail_entity_type_const.lower(),
            "tail_vocab_size": tail_vocab_size,
            "distribution": self._prepare_distrib_tensor(distrib_np),
        }

    def _create_single_relation_embedding(self, relation_name_const: str, tail_vocab_size: int) -> None:
        """Create embedding layers for a single relation."""
        embed = self._relation_embedding()
        bias = self._relation_bias(tail_vocab_size)
        
        relation_layer_name = relation_name_const.lower()
        bias_layer_name = relation_layer_name + "_bias"
        
        setattr(self, relation_layer_name, embed)
        setattr(self, bias_layer_name, bias)
        print(f"  Created embedding param '{relation_layer_name}' and bias layer '{bias_layer_name}' for relation '{relation_name_const}'")

    def _entity_embedding(self, vocab_size: int) -> nn.Embedding:
        """Create entity embedding of size [vocab_size+1, embed_size]."""
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        weight[vocab_size].fill_(0)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self) -> nn.Parameter:
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size: int) -> nn.Embedding:
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _prepare_distrib_tensor(self, distrib_np: np.ndarray) -> torch.Tensor:
        """Convert numpy distribution to tensor."""
        if not isinstance(distrib_np, np.ndarray):
            raise TypeError("Distribution must be a numpy array.")
        distrib = torch.FloatTensor(distrib_np).to(self.device)
        return distrib

    def forward(self, batch_triples: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing TransE loss.
        
        Args:
            batch_triples: Tensor of shape [batch_size, 3] containing (head, relation, tail) indices
            
        Returns:
            Computed loss tensor
        """
        loss = self.compute_loss(batch_triples)
        return loss

    def compute_loss(self, batch_triples: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge graph negative sampling loss for a batch of triples.
        
        Args:
            batch_triples: Tensor of shape [batch_size, 3] containing (head, relation, tail) indices
            
        Returns:
            Combined loss across all relation types in the batch
        """
        h_idxs = batch_triples[:, 0]
        r_idxs = batch_triples[:, 1]
        t_idxs = batch_triples[:, 2]

        loss = 0.0
        regularizations = []

        # Process each unique relation type in the batch
        unique_rel_idxs = torch.unique(r_idxs)
        for rel_idx_tensor in unique_rel_idxs:
            rel_loss, rel_embeds = self._compute_relation_loss(
                rel_idx_tensor, h_idxs, r_idxs, t_idxs
            )
            
            if rel_loss is not None:
                loss += rel_loss
                regularizations.extend(rel_embeds)

        # Apply L2 regularization
        loss = self._apply_regularization(loss, regularizations)

        # Normalize by batch size
        batch_size = batch_triples.size(0)
        if batch_size > 0:
            loss = loss / batch_size

        return loss

    def _compute_relation_loss(
        self,
        rel_idx_tensor: torch.Tensor,
        h_idxs: torch.Tensor,
        r_idxs: torch.Tensor,
        t_idxs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """Compute loss for a specific relation type."""
        rel_idx_int = rel_idx_tensor.item()
        
        # Validate relation index
        if rel_idx_int not in self.idx_to_relation_name_map:
            print(f"Warning: Relation index {rel_idx_int} not in model's idx_to_relation_name_map. Skipping.")
            return None, []

        relation_name = self.idx_to_relation_name_map[rel_idx_int]
        
        if relation_name not in self.relation_info:
            print(f"Warning: Relation '{relation_name}' not found in model's relation_info. Skipping loss calc.")
            return None, []

        # Get entity types for this relation
        head_entity_type = self._get_head_entity_type(relation_name)
        if head_entity_type is None:
            return None, []

        tail_entity_type = self.relation_info[relation_name]["tail_entity"]

        # Select batch rows for this relation
        mask = r_idxs == rel_idx_tensor
        current_h_idxs = h_idxs[mask]
        current_t_idxs = t_idxs[mask]

        # Calculate loss for this relation
        return self.neg_loss(
            head_entity_type,
            relation_name,
            tail_entity_type,
            current_h_idxs,
            current_t_idxs,
        )

    def _get_head_entity_type(self, relation_name: str) -> Optional[str]:
        """Get the head entity type for a given relation."""
        for h_type_const, rels_in_head in KG_RELATION.items():
            if relation_name in rels_in_head:
                return h_type_const.lower()
        
        print(f"Warning: Could not determine head entity type for relation '{relation_name}'. Skipping loss calculation.")
        return None

    def _apply_regularization(self, loss: torch.Tensor, regularizations: List[torch.Tensor]) -> torch.Tensor:
        """Apply L2 regularization to the loss."""
        if self.l2_lambda > 0 and regularizations:
            l2_loss = 0.0
            for term in regularizations:
                if term is not None:
                    l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss
        return loss

    def neg_loss(
        self,
        entity_head_name: str,
        relation_name: str,
        entity_tail_name: str,
        entity_head_idxs: torch.Tensor,
        entity_tail_idxs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Calculate negative sampling loss for a specific relation type.
        
        Args:
            entity_head_name: Name of head entity type
            relation_name: Name of relation
            entity_tail_name: Name of tail entity type
            entity_head_idxs: Head entity indices
            entity_tail_idxs: Tail entity indices
            
        Returns:
            Tuple of (loss, embeddings_for_regularization)
        """
        if entity_head_idxs.size(0) <= 0:
            return None, []

        # Get model components
        components = self._get_model_components(
            entity_head_name, relation_name, entity_tail_name
        )
        if components is None:
            return None, []

        entity_head_embedding, entity_tail_embedding, relation_vec, relation_bias_embedding, entity_tail_distrib = components

        try:
            # Compute positive and negative losses
            pos_loss, entity_head_vec, entity_tail_vec, example_vec, relation_bias = self._compute_positive_loss(
                entity_head_embedding, entity_tail_embedding, relation_vec, 
                relation_bias_embedding, entity_head_idxs, entity_tail_idxs
            )
            
            neg_loss = self._compute_negative_loss(
                entity_tail_embedding, entity_tail_distrib, example_vec, relation_bias
            )

            # Combine losses
            loss_per_example = pos_loss + neg_loss
            loss = loss_per_example.mean()

            return loss, [entity_head_vec, entity_tail_vec, relation_vec]

        except IndexError as e:
            self._log_index_error(e, relation_name, entity_head_idxs, entity_tail_idxs, 
                                entity_head_embedding, entity_tail_embedding, relation_bias_embedding)
            return None, []
        except Exception as e:
            print(f"Unexpected error during loss calculation for relation '{relation_name}': {e}")
            return None, []

    def _get_model_components(
        self,
        entity_head_name: str,
        relation_name: str,
        entity_tail_name: str
    ) -> Optional[Tuple]:
        """Get all model components needed for loss calculation."""
        relation_layer_name = relation_name.lower()
        bias_layer_name = relation_layer_name + "_bias"

        try:
            entity_head_embedding = getattr(self, entity_head_name)
            entity_tail_embedding = getattr(self, entity_tail_name)
            relation_vec = getattr(self, relation_layer_name)
            relation_bias_embedding = getattr(self, bias_layer_name)
            entity_tail_distrib = self.relation_info[relation_name]["distribution"]
            
            return (entity_head_embedding, entity_tail_embedding, relation_vec, 
                   relation_bias_embedding, entity_tail_distrib)
            
        except AttributeError as e:
            print(f"Error accessing layers/params for relation '{relation_name}' ({entity_head_name} -> {entity_tail_name}): {e}")
            print(f"Ensure layers '{entity_head_name}', '{entity_tail_name}', '{relation_layer_name}', '{bias_layer_name}' created correctly in __init__")
            return None
        except KeyError as e:
            print(f"Error accessing relation info for '{relation_name}': {e}")
            return None

    def _compute_positive_loss(
        self,
        entity_head_embedding: nn.Embedding,
        entity_tail_embedding: nn.Embedding,
        relation_vec: nn.Parameter,
        relation_bias_embedding: nn.Embedding,
        entity_head_idxs: torch.Tensor,
        entity_tail_idxs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute positive sample loss."""
        entity_head_vec = entity_head_embedding(entity_head_idxs)
        example_vec = entity_head_vec + relation_vec
        example_vec = example_vec.unsqueeze(2)

        entity_tail_vec = entity_tail_embedding(entity_tail_idxs)
        pos_vec = entity_tail_vec.unsqueeze(1)
        relation_bias = relation_bias_embedding(entity_tail_idxs).squeeze(1)
        
        pos_logits = torch.bmm(pos_vec, example_vec).squeeze(-1) + relation_bias
        pos_loss = -F.logsigmoid(pos_logits)

        return pos_loss, entity_head_vec, entity_tail_vec, example_vec, relation_bias

    def _compute_negative_loss(
        self,
        entity_tail_embedding: nn.Embedding,
        entity_tail_distrib: torch.Tensor,
        example_vec: torch.Tensor,
        relation_bias: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative sample loss."""
        batch_size = example_vec.size(0)
        num_neg = self.num_neg_samples
        
        # Sample negative tails
        neg_sample_idx = torch.multinomial(entity_tail_distrib, batch_size * num_neg, replacement=True)
        neg_vec = entity_tail_embedding(neg_sample_idx)
        neg_vec = neg_vec.view(batch_size, num_neg, self.embed_size)

        # Calculate negative logits
        neg_logits = torch.bmm(neg_vec, example_vec).squeeze(-1)
        neg_logits = neg_logits + relation_bias.unsqueeze(1)

        # Calculate negative loss
        neg_loss = -F.logsigmoid(-neg_logits).sum(dim=1)
        
        return neg_loss

    def _log_index_error(
        self,
        error: IndexError,
        relation_name: str,
        entity_head_idxs: torch.Tensor,
        entity_tail_idxs: torch.Tensor,
        entity_head_embedding: nn.Embedding,
        entity_tail_embedding: nn.Embedding,
        relation_bias_embedding: nn.Embedding
    ) -> None:
        """Log detailed information about index errors."""
        print(f"Error during loss calculation for relation '{relation_name}' (likely index out of bounds): {error}")
        print(
            f"  Head Indices (max): {entity_head_idxs.max() if len(entity_head_idxs) > 0 else 'N/A'}, "
            f"Head Vocab Size: {entity_head_embedding.num_embeddings}"
        )
        print(
            f"  Tail Indices (max): {entity_tail_idxs.max() if len(entity_tail_idxs) > 0 else 'N/A'}, "
            f"Tail Vocab Size: {entity_tail_embedding.num_embeddings}"
        )
        print(
            f"  Bias Indices (max): {entity_tail_idxs.max() if len(entity_tail_idxs) > 0 else 'N/A'}, "
            f"Bias Vocab Size: {relation_bias_embedding.num_embeddings}"
        )
