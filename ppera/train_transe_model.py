from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle # Needed for loading processed_dataset

from utils import *
# Removed AmazonDataset, AmazonDataLoader imports
# from data_utils import AmazonDataset, AmazonDataLoader
from transe_model import KnowledgeEmbedding # Assumes this model is adapted


logger = None

# --- NEW DATALOADER ---

class KGTrainerDataLoader:
    """
    Dataloader for KGE training.
    Yields batches of positive triples (head_idx, relation_idx, tail_idx).
    """
    def __init__(self, processed_dataset, batch_size):
        self.relations_dict = processed_dataset.get('relations', {})
        self.batch_size = batch_size

        # Create a global map from relation name to index
        self.relation_names = sorted(list(self.relations_dict.keys()))
        self.rel_to_idx = {name: i for i, name in enumerate(self.relation_names)}
        print("Relation to Index Map:", self.rel_to_idx) # Good for debugging

        # Flatten all positive triples into one list: (head_idx, rel_idx, tail_idx)
        self.triples = []
        for rel_name, head_tail_list in self.relations_dict.items():
            if rel_name in self.rel_to_idx:
                rel_idx = self.rel_to_idx[rel_name]
                for head_idx, tail_idx in head_tail_list:
                    self.triples.append((head_idx, rel_idx, tail_idx))
            else:
                 print(f"Warning: Relation '{rel_name}' found in data but not in mapped relations. Skipping.")


        if not self.triples:
             raise ValueError("No training triples found in the processed dataset!")

        self.num_triples = len(self.triples)
        print(f"KGTrainerDataLoader initialized with {self.num_triples} positive triples.")
        self.reset()

    def reset(self):
        # Shuffle triples order for each epoch
        self.indices = np.random.permutation(self.num_triples)
        self.cur_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None

        start = self.cur_idx
        end = min(self.cur_idx + self.batch_size, self.num_triples)
        batch_indices = self.indices[start:end]

        batch_triples = [self.triples[i] for i in batch_indices]

        self.cur_idx = end
        if self.cur_idx >= self.num_triples:
            self._has_next = False

        # Convert to numpy array for consistency, then to tensor later
        return np.array(batch_triples, dtype=np.int64)

    def get_num_triples_processed(self):
        return self.cur_idx

    def __len__(self):
        return (self.num_triples + self.batch_size - 1) // self.batch_size # Number of batches


# --- MODIFIED TRAIN FUNCTION ---

def train(args):
    # Load the processed dataset dictionary
    processed_dataset_file = TMP_DIR[args.dataset] + '/processed_dataset.pkl'
    print(f"Loading processed dataset from {processed_dataset_file}")
    try:
        with open(processed_dataset_file, 'rb') as f:
            processed_dataset = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Processed dataset file not found at {processed_dataset_file}")
        logger.error("Please run preprocess.py first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading processed dataset: {e}")
        sys.exit(1)

    # **Crucial Check:** Ensure distributions needed by the model exist
    if 'distributions' not in processed_dataset:
         logger.error("Error: Negative sampling distributions not found in processed_dataset.pkl.")
         logger.error("Please ensure preprocess.py calculates and saves these distributions.")
         # Example structure needed: processed_dataset['distributions']['PURCHASE'] = numpy_array
         sys.exit(1)


    # Use the new dataloader
    dataloader = KGTrainerDataLoader(processed_dataset, args.batch_size)
    # Adapt total steps calculation
    total_triples_to_train = args.epochs * dataloader.num_triples

    # Instantiate the *adapted* KnowledgeEmbedding model
    # It needs the processed_dataset for vocab sizes AND distributions
    model = KnowledgeEmbedding(processed_dataset, args).to(args.device)
    logger.info('Model Class: ' + model.__class__.__name__) # Verify model class
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()])) # Check layer names match new schema

    optimizer = optim.SGD(model.parameters(), lr=args.lr) # Or Adam, etc.
    steps = 0
    smooth_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        logger.info(f"--- Starting Epoch {epoch}/{args.epochs} ---")
        batch_num = 0
        while dataloader.has_next():
            # Calculate finished ratio based on triples processed
            triples_processed = (epoch - 1) * dataloader.num_triples + dataloader.get_num_triples_processed()
            lr_decay_factor = max(1e-4, 1.0 - triples_processed / float(total_triples_to_train + 1)) # +1 avoid zero division
            lr = args.lr * lr_decay_factor

            # Set learning rate.
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Get training batch of positive triples (h_idx, r_idx, t_idx)
            batch_triples = dataloader.get_batch()
            if batch_triples is None: continue # Should not happen with while loop logic, but safe check

            # Convert to tensor
            batch_tensor = torch.from_numpy(batch_triples).to(args.device)

            # Train model (assuming model takes [bs, 3] tensor)
            optimizer.zero_grad()
            train_loss = model(batch_tensor) # Model performs neg sampling internally
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item()

            steps += 1
            batch_num += 1
            if steps % args.steps_per_checkpoint == 0:
                avg_smooth_loss = smooth_loss / args.steps_per_checkpoint
                logger.info('Epoch: {:02d} | Batch: {:d}/{:d} | Triples: ~{:d}/{:d} | Lr: {:.5f} | Smooth loss: {:.5f}'.format(
                            epoch, batch_num, len(dataloader), triples_processed, total_triples_to_train, lr, avg_smooth_loss))
                smooth_loss = 0.0

        # Save checkpoint after each epoch
        ckpt_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch)
        logger.info(f"Saving checkpoint to {ckpt_file}")
        torch.save(model.state_dict(), ckpt_file)

# --- MODIFIED EMBEDDING EXTRACTION ---

def extract_embeddings(args):
    """Extracts embeddings using new entity/relation names."""
    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs)
    logger.info(f"Loading embeddings from final model: {model_file}")
    try:
        state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    except FileNotFoundError:
        logger.error(f"Error: Model checkpoint not found at {model_file}")
        logger.error("Training might have failed or did not complete.")
        return # Cannot extract if file doesn't exist
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {e}")
        return

    logger.info("Extracting embeddings...")
    embeds = {}
    missing_keys = []

    # Define expected layer names based on utils.py constants
    entity_layer_names = {
        USERID: 'user_id.weight', # Example: adjust based on actual layer name in model
        ITEMID: 'item_id.weight',
        TITLE: 'title.weight',
        GENRES: 'genres.weight',
        # RATING: 'rating.weight', # If rating is treated as an entity to embed
    }
    relation_param_names = {
        PURCHASE: ('purchase', 'purchase_bias.weight'), # Example: (relation_vector_name, bias_layer_name)
        MENTION: ('mentions', 'mentions_bias.weight'),
        DESCRIBED_AS: ('described_as', 'described_as_bias.weight'),
        BELONG_TO: ('belongs_to', 'belongs_to_bias.weight'),
    }

    # Extract Entity Embeddings
    for entity_const, layer_name in entity_layer_names.items():
        if layer_name in state_dict:
            # Remove the last dummy embedding
            embeds[entity_const] = state_dict[layer_name].cpu().data.numpy()[:-1]
            logger.info(f"  Extracted {entity_const} embeddings, shape: {embeds[entity_const].shape}")
        else:
            logger.warning(f"  Layer '{layer_name}' for entity '{entity_const}' not found in state_dict.")
            missing_keys.append(layer_name)

    # Extract Relation Embeddings + Biases
    for rel_const, (vec_name, bias_name) in relation_param_names.items():
        rel_vec = None
        rel_bias = None
        if vec_name in state_dict:
            rel_vec = state_dict[vec_name].cpu().data.numpy()[0] # Assuming shape [1, embed_size]
        else:
            logger.warning(f"  Parameter '{vec_name}' for relation '{rel_const}' not found in state_dict.")
            missing_keys.append(vec_name)

        if bias_name in state_dict:
             # Bias is nn.Embedding, remove last dummy bias
            rel_bias = state_dict[bias_name].cpu().data.numpy()[:-1]
        else:
            logger.warning(f"  Parameter '{bias_name}' for relation '{rel_const}' bias not found in state_dict.")
            missing_keys.append(bias_name)

        if rel_vec is not None and rel_bias is not None:
            embeds[rel_const] = (rel_vec, rel_bias)
            logger.info(f"  Extracted {rel_const} vector & bias, bias shape: {rel_bias.shape}")
        else:
            logger.warning(f"  Could not extract full data for relation '{rel_const}'.")

    if missing_keys:
        logger.error(f"Extraction incomplete. Missing state_dict keys: {missing_keys}")
        logger.error("Please ensure layer names in transe_model.py match the expected names.")
    else:
        # Save the extracted embeddings
        save_embed(args.dataset, embeds)
        logger.info("Embeddings extracted and saved successfully.")


# --- MAIN FUNCTION (Mostly Unchanged) ---

def main():
    parser = argparse.ArgumentParser()
    # Updated default dataset and help string
    parser.add_argument('--dataset', type=str, default=MOVIELENS,
                        help=f'One of {{{MOVIELENS}, {AMAZONSALES}, {POSTRECOMMENDATIONS}}}.')
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.') #default=30 fast_test=1
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.') #default=100 fast_test=10
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.') #default=5 fast_test=1
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info("----- Train KGE Start -----")
    logger.info(args)

    set_random_seed(args.seed)
    train(args)
    extract_embeddings(args)
    logger.info("----- Train KGE End -----")


if __name__ == '__main__':
    main()

