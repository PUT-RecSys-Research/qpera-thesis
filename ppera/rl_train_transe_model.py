from __future__ import absolute_import, division, print_function

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .rl_transe_model import KnowledgeEmbedding
from .rl_utils import (
    BELONG_TO,
    DESCRIBED_AS,
    GENRES,
    ITEMID,
    RATED,
    RATING,
    RATING_VALUE_FOR_ITEM,
    TITLE,
    TMP_DIR,
    USER_RATED_WITH_VALUE,
    USERID,
    WATCHED,
    get_logger,
    save_embed,
    set_random_seed,
)

logger = None


class KGTrainerDataLoader:
    """
    Dataloader for Knowledge Graph Embedding training.
    Manages batches of positive triples (head_idx, relation_idx, tail_idx).
    """

    def __init__(self, processed_dataset: Dict[str, Any], batch_size: int):
        """
        Initialize the dataloader.

        Args:
            processed_dataset: Dictionary containing relations and other data
            batch_size: Size of each training batch
        """
        self.relations_dict = processed_dataset.get("relations", {})
        self.batch_size = batch_size

        # Create relation mappings
        self.relation_names_ordered = sorted(list(self.relations_dict.keys()))
        self.rel_to_idx = {name: i for i, name in enumerate(self.relation_names_ordered)}
        self.idx_to_rel_name = {i: name for i, name in enumerate(self.relation_names_ordered)}
        print("Relation to Index Map (Dataloader):", self.rel_to_idx)

        # Build triples dataset
        self.triples = self._build_triples()
        self.num_triples = len(self.triples)

        if self.num_triples == 0:
            raise ValueError("No training triples found in the processed dataset!")

        print(f"KGTrainerDataLoader initialized with {self.num_triples} positive triples.")
        self.reset()

    def _build_triples(self) -> List[Tuple[int, int, int]]:
        """Build list of training triples from relations data."""
        triples = []

        for rel_name, head_tail_list in self.relations_dict.items():
            if rel_name in self.rel_to_idx:
                rel_idx = self.rel_to_idx[rel_name]
                for head_idx, tail_idx in head_tail_list:
                    triples.append((head_idx, rel_idx, tail_idx))
            else:
                print(f"Warning: Relation '{rel_name}' found in data but not in mapped relations. Skipping.")

        return triples

    def reset(self) -> None:
        """Reset the dataloader for a new epoch."""
        self.indices = np.random.permutation(self.num_triples)
        self.cur_idx = 0
        self._has_next = True

    def has_next(self) -> bool:
        """Check if there are more batches available."""
        return self._has_next

    def get_batch(self) -> Optional[np.ndarray]:
        """
        Get the next batch of triples.

        Returns:
            Batch of triples as numpy array or None if no more batches
        """
        if not self._has_next:
            return None

        start = self.cur_idx
        end = min(self.cur_idx + self.batch_size, self.num_triples)
        batch_indices = self.indices[start:end]

        batch_triples = [self.triples[i] for i in batch_indices]

        self.cur_idx = end
        if self.cur_idx >= self.num_triples:
            self._has_next = False

        return np.array(batch_triples, dtype=np.int64)

    def get_num_triples_processed(self) -> int:
        """Get number of triples processed so far in current epoch."""
        return self.cur_idx

    def __len__(self) -> int:
        """Get number of batches per epoch."""
        return (self.num_triples + self.batch_size - 1) // self.batch_size


def train(args: argparse.Namespace) -> None:
    """
    Main training function for TransE model.

    Args:
        args: Configuration arguments
    """
    # Load processed dataset
    processed_dataset = _load_processed_dataset(args.dataset)

    # Initialize dataloader and model
    dataloader = KGTrainerDataLoader(processed_dataset, args.batch_size)
    model = KnowledgeEmbedding(processed_dataset, dataloader.idx_to_rel_name, args).to(args.device)

    # Log model information
    logger.info("Model Class: " + model.__class__.__name__)
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))

    # Initialize optimizer and training metrics
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    total_triples_to_train = args.epochs * dataloader.num_triples

    training_state = {"steps": 0, "smooth_loss": 0.0}

    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Starting Epoch {epoch}/{args.epochs} ---")
        _train_epoch(model, optimizer, dataloader, args, epoch, total_triples_to_train, training_state)
        _save_model_checkpoint(model, epoch, args)


def _load_processed_dataset(dataset: str) -> Dict[str, Any]:
    """Load processed dataset from file."""
    processed_dataset_file = TMP_DIR[dataset] + "/processed_dataset.pkl"
    print(f"Loading processed dataset from {processed_dataset_file}")

    try:
        with open(processed_dataset_file, "rb") as f:
            processed_dataset = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Processed dataset file not found at {processed_dataset_file}")
        logger.error("Please run preprocess.py first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading processed dataset: {e}")
        sys.exit(1)

    if "distributions" not in processed_dataset:
        logger.error("Error: Negative sampling distributions not found in processed_dataset.pkl.")
        logger.error("Please ensure preprocess.py calculates and saves these distributions.")
        sys.exit(1)

    return processed_dataset


def _train_epoch(
    model: KnowledgeEmbedding,
    optimizer: torch.optim.Optimizer,
    dataloader: KGTrainerDataLoader,
    args: argparse.Namespace,
    epoch: int,
    total_triples_to_train: int,
    training_state: Dict[str, float],
) -> None:
    """Train for one epoch."""
    dataloader.reset()
    batch_num = 0

    while dataloader.has_next():
        # Update learning rate with decay
        _update_learning_rate(optimizer, dataloader, epoch, total_triples_to_train, args)

        # Get batch and train
        batch_triples = dataloader.get_batch()
        if batch_triples is None:
            continue

        # Perform training step
        loss = _training_step(model, optimizer, batch_triples, args)

        # Update training metrics
        training_state["smooth_loss"] += loss
        training_state["steps"] += 1
        batch_num += 1

        # Log progress
        if training_state["steps"] % args.steps_per_checkpoint == 0:
            _log_training_progress(epoch, batch_num, len(dataloader), dataloader, total_triples_to_train, training_state, args)


def _update_learning_rate(
    optimizer: torch.optim.Optimizer, dataloader: KGTrainerDataLoader, epoch: int, total_triples_to_train: int, args: argparse.Namespace
) -> None:
    """Update learning rate with linear decay."""
    triples_processed = (epoch - 1) * dataloader.num_triples + dataloader.get_num_triples_processed()
    lr_decay_factor = max(1e-4, 1.0 - triples_processed / float(total_triples_to_train + 1))
    lr = args.lr * lr_decay_factor

    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _training_step(model: KnowledgeEmbedding, optimizer: torch.optim.Optimizer, batch_triples: np.ndarray, args: argparse.Namespace) -> float:
    """Perform single training step."""
    batch_tensor = torch.from_numpy(batch_triples).to(args.device)

    optimizer.zero_grad()
    train_loss = model(batch_tensor)
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return train_loss.item()


def _log_training_progress(
    epoch: int,
    batch_num: int,
    total_batches: int,
    dataloader: KGTrainerDataLoader,
    total_triples_to_train: int,
    training_state: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    """Log training progress and reset smooth loss."""
    triples_processed = (epoch - 1) * dataloader.num_triples + dataloader.get_num_triples_processed()
    avg_smooth_loss = training_state["smooth_loss"] / args.steps_per_checkpoint

    # Get current learning rate
    current_lr = args.lr * max(1e-4, 1.0 - triples_processed / float(total_triples_to_train + 1))

    logger.info(
        "Epoch: {:02d} | Batch: {:d}/{:d} | Triples: ~{:d}/{:d} | Lr: {:.5f} | Smooth loss: {:.5f}".format(
            epoch,
            batch_num,
            total_batches,
            triples_processed,
            total_triples_to_train,
            current_lr,
            avg_smooth_loss,
        )
    )
    training_state["smooth_loss"] = 0.0


def _save_model_checkpoint(model: KnowledgeEmbedding, epoch: int, args: argparse.Namespace) -> None:
    """Save model checkpoint for current epoch."""
    ckpt_file = "{}/transe_model_sd_epoch_{}.ckpt".format(args.log_dir, epoch)
    logger.info(f"Saving checkpoint to {ckpt_file}")
    torch.save(model.state_dict(), ckpt_file)


def extract_embeddings(args: argparse.Namespace) -> None:
    """
    Extract embeddings from trained model and save them.

    Args:
        args: Configuration arguments
    """
    model_file = "{}/transe_model_sd_epoch_{}.ckpt".format(args.log_dir, args.epochs)
    logger.info(f"Loading embeddings from final model: {model_file}")

    # Load model state dict
    state_dict = _load_model_state_dict(model_file)
    if state_dict is None:
        return

    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeds = {}
    missing_keys = []

    # Extract entity embeddings
    _extract_entity_embeddings(state_dict, embeds, missing_keys)

    # Extract relation embeddings
    _extract_relation_embeddings(state_dict, embeds, missing_keys)

    # Check extraction completeness
    if missing_keys:
        logger.error(f"Extraction incomplete. Missing state_dict keys: {missing_keys}")
        logger.error("Please ensure layer names in transe_model.py match the expected names.")
    else:
        save_embed(args.dataset, embeds)
        logger.info("Embeddings extracted and saved successfully.")


def _load_model_state_dict(model_file: str) -> Optional[Dict]:
    """Load model state dictionary from file."""
    try:
        state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        return state_dict
    except FileNotFoundError:
        logger.error(f"Error: Model checkpoint not found at {model_file}")
        logger.error("Training might have failed or did not complete.")
        return None
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {e}")
        return None


def _extract_entity_embeddings(state_dict: Dict, embeds: Dict, missing_keys: List[str]) -> None:
    """Extract entity embeddings from state dict."""
    entity_layer_names = {
        USERID: "user_id.weight",
        ITEMID: "item_id.weight",
        TITLE: "title.weight",
        GENRES: "genres.weight",
        RATING: "rating.weight",
    }

    for entity_const, layer_name in entity_layer_names.items():
        if layer_name in state_dict:
            embeds[entity_const] = state_dict[layer_name].cpu().data.numpy()[:-1]
            logger.info(f"  Extracted {entity_const} embeddings, shape: {embeds[entity_const].shape}")
        else:
            logger.warning(f"  Layer '{layer_name}' for entity '{entity_const}' not found in state_dict.")
            missing_keys.append(layer_name)


def _extract_relation_embeddings(state_dict: Dict, embeds: Dict, missing_keys: List[str]) -> None:
    """Extract relation embeddings from state dict."""
    relation_param_names = {
        WATCHED: ("watched", "watched_bias.weight"),
        RATED: ("rated", "rated_bias.weight"),
        DESCRIBED_AS: ("described_as", "described_as_bias.weight"),
        BELONG_TO: ("belongs_to", "belongs_to_bias.weight"),
        USER_RATED_WITH_VALUE: ("user_rated_with_value", "user_rated_with_value_bias.weight"),
        RATING_VALUE_FOR_ITEM: ("rating_value_for_item", "rating_value_for_item_bias.weight"),
    }

    for rel_const, (vec_name, bias_name) in relation_param_names.items():
        rel_vec, rel_bias = _extract_single_relation(state_dict, rel_const, vec_name, bias_name, missing_keys)

        if rel_vec is not None and rel_bias is not None:
            embeds[rel_const] = (rel_vec, rel_bias)
            logger.info(f"  Extracted {rel_const} vector & bias, bias shape: {rel_bias.shape}")
        else:
            logger.warning(f"  Could not extract full data for relation '{rel_const}'.")


def _extract_single_relation(
    state_dict: Dict, rel_const: str, vec_name: str, bias_name: str, missing_keys: List[str]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract vector and bias for a single relation."""
    rel_vec = None
    rel_bias = None

    if vec_name in state_dict:
        rel_vec = state_dict[vec_name].cpu().data.numpy()[0]
    else:
        logger.warning(f"  Parameter '{vec_name}' for relation '{rel_const}' not found in state_dict.")
        missing_keys.append(vec_name)

    if bias_name in state_dict:
        rel_bias = state_dict[bias_name].cpu().data.numpy()[:-1]
    else:
        logger.warning(f"  Parameter '{bias_name}' for relation '{rel_const}' bias not found in state_dict.")
        missing_keys.append(bias_name)

    return rel_vec, rel_bias


def train_transe_model_rl(dataset: str, seed: int) -> None:
    """
    Main entry point for training TransE model.

    Args:
        dataset: Dataset name
        seed: Random seed for reproducibility
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=dataset, help="Dataset name (set automatically).")
    parser.add_argument("--name", type=str, default="train_transe_model", help="model name.")
    parser.add_argument("--seed", type=int, default=seed, help="random seed.")
    parser.add_argument("--gpu", type=str, default="1", help="gpu device.")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for adam.")
    parser.add_argument("--l2_lambda", type=float, default=0, help="l2 lambda")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Clipping gradient.")
    parser.add_argument("--embed_size", type=int, default=100, help="knowledge embedding size.")
    parser.add_argument("--num_neg_samples", type=int, default=5, help="number of negative samples.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=200, help="Number of steps for checkpoint.")
    args = parser.parse_args()

    # Set up device and directories
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    args.log_dir = "{}/{}".format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # Set up logger
    global logger
    logger = get_logger(args.log_dir + "/train_log.txt")
    logger.info("----- Train KGE Start -----")
    logger.info(args)

    # Set random seed and start training
    set_random_seed(args.seed)
    train(args)
    extract_embeddings(args)
    logger.info("----- Train KGE End -----")
