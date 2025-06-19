from rl_utils import KG_RELATION, get_entities, get_entity_tail, get_relations


class KnowledgeGraph:
    """
    Adapted KnowledgeGraph class to build from standard dictionaries.
    """

    def __init__(self, processed_dataset):
        self.G = dict()
        if not isinstance(processed_dataset, dict):
            raise TypeError("processed_dataset must be a dictionary.")
        self._load_entities(processed_dataset)
        self._load_relations(processed_dataset)
        self._clean()

    def _load_entities(self, dataset):
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

            for idx in range(vocab_size):
                self.G[entity][idx] = {r: [] for r in get_relations(entity)}
            num_nodes += vocab_size
            print(f"  Loaded {entity} with {vocab_size} nodes.")
        print(f"Total {num_nodes} nodes.")

    def _load_relations(self, dataset):
        print("Load relations...")

        relations_dict = dataset.get("relations", {})
        for relation_name, relation_data in relations_dict.items():
            print(f"  Processing relation: {relation_name} ({len(relation_data)} edges)")
            num_edges = 0
            try:
                head_entity_type = self._get_head_entity_type(relation_name)
                tail_entity_type = get_entity_tail(head_entity_type, relation_name)
            except ValueError as e:
                print(f"  Warning: Skipping relation '{relation_name}'. Error finding head/tail types: {e}")
                continue
            except KeyError as e:
                print(f"  Warning: Skipping relation '{relation_name}'. Entity type missing in KG_RELATION: {e}")
                continue

            head_map_data = self.entity_maps.get(head_entity_type, {})
            tail_map_data = self.entity_maps.get(tail_entity_type, {})
            head_vocab_size = head_map_data.get("vocab_size", 0)
            tail_vocab_size = tail_map_data.get("vocab_size", 0)

            if head_vocab_size == 0 or tail_vocab_size == 0:
                print(f"  Warning: Skipping relation '{relation_name}' due to zero vocab size for head or tail.")
                continue

            for head_idx, tail_idx in relation_data:
                if isinstance(head_idx, int) and isinstance(tail_idx, int) and 0 <= head_idx < head_vocab_size and 0 <= tail_idx < tail_vocab_size:
                    self._add_edge(head_entity_type, head_idx, relation_name, tail_entity_type, tail_idx)
                    num_edges += 2
                else:
                    print(
                        f"  Warning: Invalid indices found for relation {relation_name}: "
                        f"head={head_idx} (type={type(head_idx)}), "
                        f"tail={tail_idx} (type={type(tail_idx)}). "
                        f"Expected ints within bounds (Head Size: {head_vocab_size}, Tail Size: {tail_vocab_size})"
                    )

            print(f"    Added {num_edges} edges for {relation_name}.")

    def _get_head_entity_type(self, relation_name):
        for head_type, relations in KG_RELATION.items():
            if relation_name in relations:
                return head_type
        raise ValueError(f"Could not determine head entity type for relation: {relation_name}")

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        if etype1 in self.G and etype2 in self.G and eid1 in self.G[etype1] and eid2 in self.G[etype2]:
            if relation in self.G[etype1][eid1]:
                self.G[etype1][eid1][relation].append(eid2)
            if relation in self.G[etype2][eid2]:
                self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print("Cleaning graph: Removing duplicate edges...")
        for etype in self.G:
            for eid in self.G.get(etype, {}):
                for r in self.G[etype][eid]:
                    unique_neighbors = list(set(self.G[etype][eid][r]))
                    self.G[etype][eid][r] = sorted(unique_neighbors)

    def compute_degrees(self):
        print("Compute node degrees...")
        self.degrees = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G.get(etype, {}):
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count
        print("Node degrees computed.")

    def get(self, eh_type, eh_id=None, relation=None):
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

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)
