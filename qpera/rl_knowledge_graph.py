from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Optional, Union

from .rl_utils import KG_RELATION, get_entities, get_entity_tail, get_relations


class KnowledgeGraph:
    """
    Knowledge Graph class for building and managing entity-relation structures.

    Builds a graph from processed dataset dictionaries containing entity mappings
    and relation data. Supports efficient querying of entities, relations, and
    their connections for recommendation algorithms.
    """

    def __init__(self, processed_dataset: Dict[str, Any]):
        """
        Initialize knowledge graph from processed dataset.

        Args:
            processed_dataset: Dictionary containing entity_maps and relations

        Raises:
            TypeError: If processed_dataset is not a dictionary
        """
        self.G = dict()
        if not isinstance(processed_dataset, dict):
            raise TypeError("processed_dataset must be a dictionary.")

        self._load_entities(processed_dataset)
        self._load_relations(processed_dataset)
        self._clean()

    def _load_entities(self, dataset: Dict[str, Any]) -> None:
        """Load entities and initialize graph structure."""
        print("Load entities...")

        self.entity_maps = dataset.get("entity_maps", {})
        num_nodes = 0

        for entity in get_entities():
            self.G[entity] = {}
            entity_map_data = self.entity_maps.get(entity, {})
            vocab_size = entity_map_data.get("vocab_size", 0)

            if vocab_size == 0:
                print(f"  Warning: Vocab size is 0 for entity '{entity}'. Skipping graph initialization for it.")
                continue

            # Initialize nodes with empty relation lists
            for idx in range(vocab_size):
                self.G[entity][idx] = {r: [] for r in get_relations(entity)}

            num_nodes += vocab_size
            print(f"  Loaded {entity} with {vocab_size} nodes.")

        print(f"Total {num_nodes} nodes.")

    def _load_relations(self, dataset: Dict[str, Any]) -> None:
        """Load relations and build edges in the graph."""
        print("Load relations...")

        relations_dict = dataset.get("relations", {})

        for relation_name, relation_data in relations_dict.items():
            print(f"  Processing relation: {relation_name} ({len(relation_data)} edges)")

            # Validate relation and get entity types
            entity_types = self._validate_relation(relation_name)
            if entity_types is None:
                continue

            head_entity_type, tail_entity_type = entity_types

            # Check vocabulary sizes
            if not self._validate_vocab_sizes(relation_name, head_entity_type, tail_entity_type):
                continue

            # Add edges for this relation
            num_edges = self._add_relation_edges(relation_name, relation_data, head_entity_type, tail_entity_type)
            print(f"    Added {num_edges} edges for {relation_name}.")

    def _validate_relation(self, relation_name: str) -> Optional[tuple]:
        """Validate relation and return head/tail entity types."""
        try:
            head_entity_type = self._get_head_entity_type(relation_name)
            tail_entity_type = get_entity_tail(head_entity_type, relation_name)
            return head_entity_type, tail_entity_type
        except ValueError as e:
            print(f"  Warning: Skipping relation '{relation_name}'. Error finding head/tail types: {e}")
            return None
        except KeyError as e:
            print(f"  Warning: Skipping relation '{relation_name}'. Entity type missing in KG_RELATION: {e}")
            return None

    def _validate_vocab_sizes(self, relation_name: str, head_entity_type: str, tail_entity_type: str) -> bool:
        """Validate that entity types have non-zero vocabulary sizes."""
        head_map_data = self.entity_maps.get(head_entity_type, {})
        tail_map_data = self.entity_maps.get(tail_entity_type, {})
        head_vocab_size = head_map_data.get("vocab_size", 0)
        tail_vocab_size = tail_map_data.get("vocab_size", 0)

        if head_vocab_size == 0 or tail_vocab_size == 0:
            print(f"  Warning: Skipping relation '{relation_name}' due to zero vocab size for head or tail.")
            return False

        return True

    def _add_relation_edges(self, relation_name: str, relation_data: List[tuple], head_entity_type: str, tail_entity_type: str) -> int:
        """Add edges for a specific relation."""
        head_vocab_size = self.entity_maps[head_entity_type]["vocab_size"]
        tail_vocab_size = self.entity_maps[tail_entity_type]["vocab_size"]
        num_edges = 0

        for head_idx, tail_idx in relation_data:
            if self._validate_edge_indices(head_idx, tail_idx, head_vocab_size, tail_vocab_size, relation_name):
                self._add_edge(head_entity_type, head_idx, relation_name, tail_entity_type, tail_idx)
                num_edges += 2  # Bidirectional edge

        return num_edges

    def _validate_edge_indices(self, head_idx: Any, tail_idx: Any, head_vocab_size: int, tail_vocab_size: int, relation_name: str) -> bool:
        """Validate that edge indices are valid integers within bounds."""
        if not (isinstance(head_idx, int) and isinstance(tail_idx, int)):
            print(
                f"  Warning: Invalid indices found for relation {relation_name}: "
                f"head={head_idx} (type={type(head_idx)}), "
                f"tail={tail_idx} (type={type(tail_idx)}). Expected integers."
            )
            return False

        if not (0 <= head_idx < head_vocab_size and 0 <= tail_idx < tail_vocab_size):
            print(
                f"  Warning: Out of bounds indices for relation {relation_name}: "
                f"head={head_idx}, tail={tail_idx}. "
                f"Expected within bounds (Head Size: {head_vocab_size}, Tail Size: {tail_vocab_size})"
            )
            return False

        return True

    def _get_head_entity_type(self, relation_name: str) -> str:
        """Find the head entity type for a given relation."""
        for head_type, relations in KG_RELATION.items():
            if relation_name in relations:
                return head_type
        raise ValueError(f"Could not determine head entity type for relation: {relation_name}")

    def _add_edge(self, etype1: str, eid1: int, relation: str, etype2: str, eid2: int) -> None:
        """Add bidirectional edge between two entities."""
        # Validate entities exist in graph
        if not self._entities_exist(etype1, eid1, etype2, eid2):
            return

        # Add edge from entity1 to entity2
        if relation in self.G[etype1][eid1]:
            self.G[etype1][eid1][relation].append(eid2)

        # Add reverse edge from entity2 to entity1
        if relation in self.G[etype2][eid2]:
            self.G[etype2][eid2][relation].append(eid1)

    def _entities_exist(self, etype1: str, eid1: int, etype2: str, eid2: int) -> bool:
        """Check if both entities exist in the graph."""
        return etype1 in self.G and etype2 in self.G and eid1 in self.G[etype1] and eid2 in self.G[etype2]

    def _clean(self) -> None:
        """Remove duplicate edges and sort neighbor lists."""
        print("Cleaning graph: Removing duplicate edges...")

        for etype in self.G:
            for eid in self.G.get(etype, {}):
                for relation in self.G[etype][eid]:
                    # Remove duplicates and sort
                    unique_neighbors = list(set(self.G[etype][eid][relation]))
                    self.G[etype][eid][relation] = sorted(unique_neighbors)

    def compute_degrees(self) -> None:
        """Compute and store the degree of each node."""
        print("Compute node degrees...")
        self.degrees = {}

        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G.get(etype, {}):
                # Count total neighbors across all relations
                total_degree = sum(len(self.G[etype][eid][relation]) for relation in self.G[etype][eid])
                self.degrees[etype][eid] = total_degree

        print("Node degrees computed.")

    def get_node_degree(self, entity_type: str, entity_id: int) -> int:
        """
        Get the degree of a specific node.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            Node degree (0 if node doesn't exist)
        """
        if not hasattr(self, "degrees"):
            self.compute_degrees()

        return self.degrees.get(entity_type, {}).get(entity_id, 0)

    def get_entity_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about entities in the graph.

        Returns:
            Dictionary with entity statistics
        """
        stats = {}

        for entity_type in self.G:
            num_nodes = len(self.G[entity_type])
            total_edges = 0

            for eid in self.G[entity_type]:
                for relation in self.G[entity_type][eid]:
                    total_edges += len(self.G[entity_type][eid][relation])

            stats[entity_type] = {"num_nodes": num_nodes, "total_edges": total_edges}

        return stats

    def get_relation_stats(self) -> Dict[str, int]:
        """
        Get statistics about relations in the graph.

        Returns:
            Dictionary mapping relation names to edge counts
        """
        relation_counts = {}

        for entity_type in self.G:
            for eid in self.G[entity_type]:
                for relation in self.G[entity_type][eid]:
                    if relation not in relation_counts:
                        relation_counts[relation] = 0
                    relation_counts[relation] += len(self.G[entity_type][eid][relation])

        # Divide by 2 since edges are bidirectional
        return {rel: count // 2 for rel, count in relation_counts.items()}

    def get(self, eh_type: str, eh_id: Optional[int] = None, relation: Optional[str] = None) -> Union[Dict, List]:
        """
        Get data from the graph with optional filtering.

        Args:
            eh_type: Entity type
            eh_id: Entity ID (optional)
            relation: Relation name (optional)

        Returns:
            Graph data filtered by the provided parameters
        """
        data = self.G

        if eh_type is not None:
            data = data.get(eh_type, {})

        if eh_id is not None:
            try:
                eh_id_int = int(eh_id)
                data = data.get(eh_id_int, {})
            except (ValueError, TypeError):
                data = {}

        if relation is not None:
            data = data.get(relation, [])

        return data

    def __call__(self, eh_type: str, eh_id: Optional[int] = None, relation: Optional[str] = None) -> Union[Dict, List]:
        """
        Callable interface for graph querying.

        Args:
            eh_type: Entity type
            eh_id: Entity ID (optional)
            relation: Relation name (optional)

        Returns:
            Graph data filtered by the provided parameters
        """
        return self.get(eh_type, eh_id, relation)

    def print_stats(self) -> None:
        """Print comprehensive graph statistics."""
        print("\n" + "=" * 50)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("=" * 50)

        # Entity statistics
        entity_stats = self.get_entity_stats()
        print("\nEntity Statistics:")
        for entity_type, stats in entity_stats.items():
            print(f"  {entity_type}: {stats['num_nodes']} nodes, {stats['total_edges']} edges")

        # Relation statistics
        relation_stats = self.get_relation_stats()
        print("\nRelation Statistics:")
        for relation, count in sorted(relation_stats.items()):
            print(f"  {relation}: {count} edges")

        # Total statistics
        total_nodes = sum(stats["num_nodes"] for stats in entity_stats.values())
        total_edges = sum(relation_stats.values())
        print(f"\nTotal: {total_nodes} nodes, {total_edges} unique edges")
        print("=" * 50)
