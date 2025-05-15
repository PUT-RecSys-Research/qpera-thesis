from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_utils import *


class KnowledgeEmbedding(nn.Module):
    def __init__(self, processed_dataset, idx_to_relation_name_map, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.idx_to_relation_name_map = idx_to_relation_name_map

        if not isinstance(processed_dataset, dict):
            raise TypeError("KnowledgeEmbedding expects processed_dataset to be a dictionary.")

        entity_maps = processed_dataset.get('entity_maps', {})
        distributions = processed_dataset.get('distributions', {})

        self.entity_names = get_entities()
        for entity_name in self.entity_names:
            map_data = entity_maps.get(entity_name, {})
            vocab_size = map_data.get('vocab_size', 0)
            if vocab_size == 0:
                 print(f"Warning: Vocab size is 0 for entity '{entity_name}'. Skipping embedding creation.")
                 continue
            embed = self._entity_embedding(vocab_size)
            layer_name = entity_name.lower()
            setattr(self, layer_name, embed)
            print(f"  Created embedding layer '{layer_name}' for entity '{entity_name}' with size {vocab_size+1}")

        self.relation_info = {}
        # self.relation_names = []

        for head_entity_type, relations_in_head in KG_RELATION.items():
            for relation_name_const in relations_in_head:
                if relation_name_const in self.relation_info: continue

                tail_entity_type_const = get_entity_tail(head_entity_type, relation_name_const)
                tail_map_data = entity_maps.get(tail_entity_type_const, {})
                tail_vocab_size = tail_map_data.get('vocab_size', 0)
                distrib_np = distributions.get(relation_name_const, None)

                if tail_vocab_size == 0:
                    print(f"Warning: Tail vocab size is 0 for relation '{relation_name_const}'. Skipping relation layer creation.")
                    continue
                if distrib_np  is None:
                    print(f"Warning: Distribution not found for relation '{relation_name_const}'. Skipping relation layer creation.")
                    continue

                self.relation_info[relation_name_const] = {
                    'head_entity': head_entity_type.lower(),
                    'tail_entity': tail_entity_type_const.lower(),
                    'tail_vocab_size': tail_vocab_size,
                    'distribution': self._prepare_distrib_tensor(distrib_np)
                }

                embed = self._relation_embedding()
                bias = self._relation_bias(tail_vocab_size)
                relation_layer_name = relation_name_const.lower()
                bias_layer_name = relation_layer_name + '_bias'
                setattr(self, relation_layer_name, embed)
                setattr(self, bias_layer_name, bias)
                print(f"  Created embedding param '{relation_layer_name}' and bias layer '{bias_layer_name}' for relation '{relation_name_const}'")


    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size]."""
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        weight[vocab_size].fill_(0)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _prepare_distrib_tensor(self, distrib_np):
        """Convert numpy distribution to tensor."""
        if not isinstance(distrib_np, np.ndarray):
             raise TypeError("Distribution must be a numpy array.")
        distrib = torch.FloatTensor(distrib_np).to(self.device)
        return distrib

    def forward(self, batch_triples):
        loss = self.compute_loss(batch_triples)
        return loss

    def compute_loss(self, batch_triples):
        """Compute knowledge graph negative sampling loss for a batch of triples."""
        h_idxs = batch_triples[:, 0]
        r_idxs = batch_triples[:, 1]
        t_idxs = batch_triples[:, 2]

        loss = 0.0
        regularizations = []

        unique_rel_idxs = torch.unique(r_idxs)
        for rel_idx_tensor in unique_rel_idxs:
            rel_idx_int = rel_idx_tensor.item()
            if rel_idx_int not in self.idx_to_relation_name_map:
                print(f"Warning: Relation index {rel_idx_int} from batch not in model's idx_to_relation_name_map. Skipping.")
                continue
            relation_name = self.idx_to_relation_name_map[rel_idx_int]

            if relation_name not in self.relation_info:
                print(f"Warning: Relation '{relation_name}' (from index {rel_idx_int}) not found in model's self.relation_info. Skipping loss calculation.")
                continue

            head_entity_type_const = None
            for h_type_const, rels_in_head in KG_RELATION.items():
                if relation_name in rels_in_head:
                    head_entity_type_const = h_type_const
                    break
            if head_entity_type_const is None:
                 print(f"Warning: Could not determine head entity type for relation '{relation_name}'. Skipping loss calculation.")
                 continue
            
            head_entity_type_str_lower = head_entity_type_const.lower()

            tail_entity_type_str_lower = self.relation_info[relation_name]['tail_entity']

            # Select rows in the batch corresponding to this relation
            mask = (r_idxs == rel_idx_tensor) # Use tensor for mask
            current_h_idxs = h_idxs[mask]
            current_t_idxs = t_idxs[mask]

            # Calculate loss for this specific relation's batch
            rel_loss, rel_embeds = self.neg_loss(
                head_entity_type_str_lower,
                relation_name,
                tail_entity_type_str_lower,
                current_h_idxs,
                current_t_idxs
            )

            if rel_loss is not None:
                loss += rel_loss
                regularizations.extend(rel_embeds)


        # l2 regularization
        if self.l2_lambda > 0 and regularizations:
            l2_loss = 0.0
            for term in regularizations:
                if term is not None:
                     l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        batch_size = batch_triples.size(0)
        if batch_size > 0:
             loss = loss / batch_size

        return loss


    def neg_loss(self, entity_head_name, relation_name, entity_tail_name,
                 entity_head_idxs, entity_tail_idxs):
        """Calculates negative sampling loss for a specific relation type."""
        relation_layer_name = relation_name.lower() # Convert constant to lowercase for layer name
        bias_layer_name = relation_layer_name + '_bias'

        try:
            entity_head_embedding = getattr(self, entity_head_name) # Uses "user_id"
            entity_tail_embedding = getattr(self, entity_tail_name) # Uses "item_id"
            relation_vec = getattr(self, relation_layer_name)      # Uses "watched"
            relation_bias_embedding = getattr(self, bias_layer_name) # Uses "watched_bias"
            entity_tail_distrib = self.relation_info[relation_name]['distribution']
        except AttributeError as e:
             print(f"Error accessing layers/params for relation '{relation_name}' ({entity_head_name} -> {entity_tail_name}): {e}")
             print(f"  Ensure layers '{entity_head_name}', '{entity_tail_name}', '{relation_layer_name}', '{bias_layer_name}' were created correctly in __init__.")
             return None, [] 
        except KeyError as e:
             print(f"Error accessing relation info for '{relation_name}': {e}")
             return None, []

        if entity_head_idxs.size(0) <= 0:
            return None, []

        batch_size = entity_head_idxs.size(0)

        try:
            entity_head_vec = entity_head_embedding(entity_head_idxs)
            example_vec = entity_head_vec + relation_vec
            example_vec = example_vec.unsqueeze(2)

            # Positive sample scoring
            entity_tail_vec = entity_tail_embedding(entity_tail_idxs)
            pos_vec = entity_tail_vec.unsqueeze(1)
            relation_bias = relation_bias_embedding(entity_tail_idxs).squeeze(1)
            pos_logits = torch.bmm(pos_vec, example_vec).squeeze(-1) + relation_bias
            pos_loss = -F.logsigmoid(pos_logits)

            # Negative sample scoring
            # Sample negative tails using the pre-calculated distribution tensor
            num_neg = self.num_neg_samples
            neg_sample_idx = torch.multinomial(entity_tail_distrib, batch_size * num_neg, replacement=True)
            neg_vec = entity_tail_embedding(neg_sample_idx)
            neg_vec = neg_vec.view(batch_size, num_neg, self.embed_size)

            # Calculate dot product with predicted vector (example_vec)
            neg_logits = torch.bmm(neg_vec, example_vec).squeeze(-1)

            # Add bias of the *positive* tail entity
            neg_logits = neg_logits + relation_bias.unsqueeze(1)

            # Calculate negative loss using log-sigmoid
            neg_loss = -F.logsigmoid(-neg_logits).sum(dim=1)

            # Combine losses
            # Taking mean over the batch dimension here for this relation type
            loss_per_example = pos_loss + neg_loss
            loss = loss_per_example.mean()

            # Return embeddings involved for regularization
            return loss, [entity_head_vec, entity_tail_vec, relation_vec]

        except IndexError as e:
             print(f"Error during loss calculation for relation '{relation_name}' (likely index out of bounds): {e}")
             print(f"  Head Indices (max): {entity_head_idxs.max() if len(entity_head_idxs)>0 else 'N/A'}, Head Vocab Size: {entity_head_embedding.num_embeddings}")
             print(f"  Tail Indices (max): {entity_tail_idxs.max() if len(entity_tail_idxs)>0 else 'N/A'}, Tail Vocab Size: {entity_tail_embedding.num_embeddings}")
             print(f"  Bias Indices (max): {entity_tail_idxs.max() if len(entity_tail_idxs)>0 else 'N/A'}, Bias Vocab Size: {relation_bias_embedding.num_embeddings}")
             return None, []
        except Exception as e:
             print(f"Unexpected error during loss calculation for relation '{relation_name}': {e}")
             return None, []


