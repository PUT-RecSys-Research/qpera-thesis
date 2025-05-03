from __future__ import absolute_import, division, print_function

# Removed EasyDict import - we will use standard dicts where needed
# from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
# Removed AmazonDataset import as it's no longer used directly
# from data_utils import AmazonDataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, processed_dataset, args): # Renamed dataset -> processed_dataset for clarity
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        # Ensure processed_dataset is a dict
        if not isinstance(processed_dataset, dict):
            raise TypeError("KnowledgeEmbedding expects processed_dataset to be a dictionary.")

        entity_maps = processed_dataset.get('entity_maps', {})
        distributions = processed_dataset.get('distributions', {}) # Get distributions dict

        # Initialize entity embeddings using dictionary access
        # Use constants from utils.py for keys
        self.entity_names = get_entities() # Get list like [USERID, ITEMID, TITLE, GENRES]
        for entity_name in self.entity_names:
            map_data = entity_maps.get(entity_name, {})
            vocab_size = map_data.get('vocab_size', 0)
            if vocab_size == 0:
                 print(f"Warning: Vocab size is 0 for entity '{entity_name}'. Skipping embedding creation.")
                 continue
            embed = self._entity_embedding(vocab_size)
            # Layer name should ideally be lowercase version of constant for consistency
            layer_name = entity_name.lower() # e.g., 'userid', 'itemid'
            setattr(self, layer_name, embed)
            print(f"  Created embedding layer '{layer_name}' for entity '{entity_name}' with size {vocab_size+1}")

        # Initialize relation embeddings and relation biases using dictionary access
        # Use relation constants from utils.py for keys
        # Need a way to map relation names to info (tail type) and distributions
        self.relation_info = {} # Store info for later use
        self.relation_names = [] # Keep track of relations we actually create layers for

        # Iterate through the relations defined in the schema
        for head_entity_type, relations in KG_RELATION.items():
            for relation_name in relations:
                 if relation_name in self.relation_info: continue # Already processed

                 tail_entity_type = get_entity_tail(head_entity_type, relation_name)
                 tail_map_data = entity_maps.get(tail_entity_type, {})
                 tail_vocab_size = tail_map_data.get('vocab_size', 0)
                 distrib = distributions.get(relation_name, None)

                 if tail_vocab_size == 0:
                      print(f"Warning: Tail vocab size is 0 for relation '{relation_name}'. Skipping relation layer creation.")
                      continue
                 if distrib is None:
                     print(f"Warning: Distribution not found for relation '{relation_name}'. Skipping relation layer creation.")
                     continue

                 # Store info
                 self.relation_info[relation_name] = {
                     'tail_entity': tail_entity_type.lower(), # Store lowercase name for consistency
                     'tail_vocab_size': tail_vocab_size,
                     'distribution': self._prepare_distrib_tensor(distrib) # Convert numpy to tensor
                 }

                 # Create layers
                 embed = self._relation_embedding()
                 bias = self._relation_bias(tail_vocab_size)

                 # Layer names should be lowercase version of constant
                 relation_layer_name = relation_name.lower() # e.g., 'purchase', 'mentions'
                 bias_layer_name = relation_layer_name + '_bias' # e.g., 'purchase_bias'

                 setattr(self, relation_layer_name, embed)
                 setattr(self, bias_layer_name, bias)
                 self.relation_names.append(relation_name) # Add to list of processed relations
                 print(f"  Created embedding param '{relation_layer_name}' and bias layer '{bias_layer_name}' for relation '{relation_name}'")


    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size]."""
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        # Need to handle padding index explicitly if using 0 for it
        weight[vocab_size].fill_(0) # Ensure last row (padding index if -1 maps to vocab_size) is zeros
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
        # Initialize biases to zero
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    # Renamed _make_distrib to avoid confusion, now just converts numpy to tensor
    def _prepare_distrib_tensor(self, distrib_np):
        """Convert numpy distribution to tensor."""
        # Assuming distrib_np is already the final probability distribution
        if not isinstance(distrib_np, np.ndarray):
             raise TypeError("Distribution must be a numpy array.")
        distrib = torch.FloatTensor(distrib_np).to(self.device)
        return distrib

    def forward(self, batch_triples):
        # batch_triples: tensor of shape [batch_size, 3] containing (h_idx, r_idx, t_idx)
        loss = self.compute_loss(batch_triples)
        return loss

    def compute_loss(self, batch_triples):
        """Compute knowledge graph negative sampling loss for a batch of triples."""
        # Split the batch into head, relation, tail indices
        h_idxs = batch_triples[:, 0]
        r_idxs = batch_triples[:, 1] # These are indices mapping to relation NAMES
        t_idxs = batch_triples[:, 2]

        loss = 0.0
        regularizations = [] # Collect embeddings for optional L2 reg

        # Map relation indices back to names to call neg_loss correctly
        # Need the inverse map from the dataloader or create it here
        # Assuming KGTrainerDataLoader created self.relation_names
        # This assumes train.py passes this map or we recreate it (less ideal)
        # Let's recreate for now, though passing is better
        # TODO: Consider passing relation_names list or idx_to_rel map from dataloader via args or dataset dict
        temp_idx_to_rel = {i: name for i, name in enumerate(sorted(self.relation_info.keys()))}


        # Process loss for each relation type present in the batch
        unique_rel_idxs = torch.unique(r_idxs)
        for rel_idx in unique_rel_idxs:
            rel_idx_int = rel_idx.item() # Convert tensor to int
            if rel_idx_int not in temp_idx_to_rel:
                 print(f"Warning: Relation index {rel_idx_int} out of bounds. Skipping.")
                 continue

            relation_name = temp_idx_to_rel[rel_idx_int]

            # Find head entity type based on the relation name
            head_entity_type = None
            for h_type, rels in KG_RELATION.items():
                if relation_name in rels:
                    head_entity_type = h_type.lower() # Use lowercase for layer names
                    break
            if head_entity_type is None:
                 print(f"Warning: Could not determine head entity type for relation '{relation_name}'. Skipping loss calculation.")
                 continue

            # Get tail entity type from stored info
            tail_entity_type = self.relation_info[relation_name]['tail_entity'] # Already lowercase

            # Select rows in the batch corresponding to this relation
            mask = (r_idxs == rel_idx)
            current_h_idxs = h_idxs[mask]
            current_t_idxs = t_idxs[mask]

            # Calculate loss for this specific relation's batch
            rel_loss, rel_embeds = self.neg_loss(
                head_entity_type,    # e.g., 'userid'
                relation_name,       # e.g., 'purchase'
                tail_entity_type,    # e.g., 'itemid'
                current_h_idxs,
                current_t_idxs
            )

            if rel_loss is not None:
                # Accumulate loss (consider weighting different relations?)
                loss += rel_loss
                regularizations.extend(rel_embeds)


        # l2 regularization
        if self.l2_lambda > 0 and regularizations:
            l2_loss = 0.0
            for term in regularizations:
                if term is not None: # Ensure embeddings were returned
                     l2_loss += torch.norm(term)**2 # Square norm for L2
            # Normalize L2 loss by number of terms? Or apply lambda directly? Applying directly.
            loss += self.l2_lambda * l2_loss

        # Average loss over the batch size
        batch_size = batch_triples.size(0)
        if batch_size > 0:
             loss = loss / batch_size # Average the summed loss

        return loss


    def neg_loss(self, entity_head_name, relation_name, entity_tail_name,
                 entity_head_idxs, entity_tail_idxs):
        """Calculates negative sampling loss for a specific relation type."""
        # Args now use string names for entities/relations

        # Get layers/parameters using lowercase names derived from constants
        entity_head_layer_name = entity_head_name.lower()
        entity_tail_layer_name = entity_tail_name.lower()
        relation_layer_name = relation_name.lower()
        bias_layer_name = relation_layer_name + '_bias'

        try:
            entity_head_embedding = getattr(self, entity_head_layer_name)
            entity_tail_embedding = getattr(self, entity_tail_layer_name)
            relation_vec = getattr(self, relation_layer_name)
            relation_bias_embedding = getattr(self, bias_layer_name)
            # Get distribution tensor from stored info
            entity_tail_distrib = self.relation_info[relation_name]['distribution']
        except AttributeError as e:
             print(f"Error accessing layers/params for relation '{relation_name}' ({entity_head_name} -> {entity_tail_name}): {e}")
             print(f"  Ensure layers '{entity_head_layer_name}', '{entity_tail_layer_name}', '{relation_layer_name}', '{bias_layer_name}' were created correctly in __init__.")
             return None, [] # Cannot calculate loss if layers missing
        except KeyError as e:
             print(f"Error accessing relation info for '{relation_name}': {e}")
             return None, []


        # --- kg_neg_loss logic moved directly here ---
        # We no longer need to filter for -1 indices, as dataloader gives valid triples

        if entity_head_idxs.size(0) <= 0: # Check if any examples remain for this relation
            return None, []

        batch_size = entity_head_idxs.size(0)

        try:
            entity_head_vec = entity_head_embedding(entity_head_idxs)  # [batch_size, embed_size]
            # TransE prediction
            example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
            example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

            # Positive sample scoring
            entity_tail_vec = entity_tail_embedding(entity_tail_idxs)  # [batch_size, embed_size]
            pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
            relation_bias = relation_bias_embedding(entity_tail_idxs).squeeze(1)  # [batch_size]
            pos_logits = torch.bmm(pos_vec, example_vec).squeeze(-1) + relation_bias  # [batch_size] # Fixed squeeze dim
            pos_loss = -F.logsigmoid(pos_logits) # More stable than sigmoid().log() # [batch_size]

            # Negative sample scoring
            # Sample negative tails using the pre-calculated distribution tensor
            num_neg = self.num_neg_samples
            neg_sample_idx = torch.multinomial(entity_tail_distrib, batch_size * num_neg, replacement=True)
            neg_vec = entity_tail_embedding(neg_sample_idx)  # [batch_size * num_neg, embed_size]
            neg_vec = neg_vec.view(batch_size, num_neg, self.embed_size) # [batch_size, num_neg, embed_size]

            # Calculate dot product with predicted vector (example_vec)
            # example_vec is [batch_size, embed_size, 1]
            # neg_vec is [batch_size, num_neg, embed_size] -> transpose last two dims
            neg_logits = torch.bmm(neg_vec, example_vec).squeeze(-1) # [batch_size, num_neg]

            # Add bias of the *positive* tail entity
            neg_logits = neg_logits + relation_bias.unsqueeze(1)  # [batch_size, num_neg]

            # Calculate negative loss using log-sigmoid
            neg_loss = -F.logsigmoid(-neg_logits).sum(dim=1)  # Sum over negative samples # [batch_size]

            # Combine losses
            # Taking mean over the batch dimension here for this relation type
            loss_per_example = pos_loss + neg_loss
            loss = loss_per_example.mean()

            # Return embeddings involved for regularization
            # Note: neg_vec needs reshaping if used directly for L2
            return loss, [entity_head_vec, entity_tail_vec, relation_vec] # Don't return neg_vec directly for L2 for now

        except IndexError as e:
             print(f"Error during loss calculation for relation '{relation_name}' (likely index out of bounds): {e}")
             # Print relevant indices to debug
             print(f"  Head Indices (max): {entity_head_idxs.max() if len(entity_head_idxs)>0 else 'N/A'}, Head Vocab Size: {entity_head_embedding.num_embeddings}")
             print(f"  Tail Indices (max): {entity_tail_idxs.max() if len(entity_tail_idxs)>0 else 'N/A'}, Tail Vocab Size: {entity_tail_embedding.num_embeddings}")
             print(f"  Bias Indices (max): {entity_tail_idxs.max() if len(entity_tail_idxs)>0 else 'N/A'}, Bias Vocab Size: {relation_bias_embedding.num_embeddings}")
             return None, []
        except Exception as e:
             print(f"Unexpected error during loss calculation for relation '{relation_name}': {e}")
             return None, []


