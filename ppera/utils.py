from __future__ import absolute_import, division, print_function

import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch


# Dataset names.
MOVIELENS = 'movielens'
AMAZONSALES = 'amazonsales'
POSTRECOMMENDATIONS = 'PostRecommendations'

# Dataset directories.
DATASET_DIR = {
    MOVIELENS: './datasets/movielens',
    AMAZONSALES: './datasets/amazonsales',
    POSTRECOMMENDATIONS: './datasets/PostRecommendations',
}

# Model result directories.
TMP_DIR = {
    MOVIELENS: './tmp/Movielens',
    AMAZONSALES: './tmp/AmazonSales',
    POSTRECOMMENDATIONS: './tmp/PostRecommendations',
}

# Label files.
LABELS = {
    MOVIELENS: (TMP_DIR[MOVIELENS] + '/train_label.pkl', TMP_DIR[MOVIELENS] + '/test_label.pkl'),
    AMAZONSALES: (TMP_DIR[AMAZONSALES] + '/train_label.pkl', TMP_DIR[AMAZONSALES] + '/test_label.pkl'),
    POSTRECOMMENDATIONS: (TMP_DIR[POSTRECOMMENDATIONS] + '/train_label.pkl', TMP_DIR[POSTRECOMMENDATIONS] + '/test_label.pkl'),
}


# Entities
USERID = 'user_id'
ITEMID = 'item_id'
TITLE = 'title'
GENRES = 'genres'
RATING = 'rating'


# Relations
PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
BELONG_TO = 'belongs_to'
SELF_LOOP = 'self_loop'  # only for kg env

KG_RELATION = {
    USERID: {
        PURCHASE: ITEMID,
        MENTION: TITLE,
    },
    TITLE: {
        MENTION: USERID,
        DESCRIBED_AS: ITEMID,
    },
    ITEMID: {
        PURCHASE: USERID,
        DESCRIBED_AS: TITLE,
        BELONG_TO: GENRES,
    },
    GENRES: {
        BELONG_TO: ITEMID,
    }
}


PATH_PATTERN = {
    # length = 3
    1: ((None, USERID), (MENTION, TITLE), (DESCRIBED_AS, ITEMID)),
    # length = 4
    11: ((None, USERID), (PURCHASE, ITEMID), (PURCHASE, USERID), (PURCHASE, ITEMID)),
    12: ((None, USERID), (PURCHASE, ITEMID), (DESCRIBED_AS, TITLE), (DESCRIBED_AS, ITEMID)),
    13: ((None, USERID), (PURCHASE, ITEMID), (BELONG_TO, GENRES), (BELONG_TO, ITEMID)),
    14: ((None, USERID), (MENTION, TITLE), (MENTION, USERID), (PURCHASE, ITEMID)),
}


def get_entities():
    return list(KG_RELATION.keys())


def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

