import pickle
import os
import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import torch
import faiss


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def transform(embeddings, noise_level, linear_transform):
    if noise_level > 0:
        embeddings += noise_level * torch.randn(embeddings.shape, dtype=torch.float32).numpy()
    if linear_transform:
        embeddings = embeddings * -2.6
    return embeddings

def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')

parser = ArgumentParser()
parser.add_argument('--query_reps', required=True)
parser.add_argument('--passage_reps', required=True)
parser.add_argument('--query_save_to', required=True)
parser.add_argument('--passage_save_to', required=True)
parser.add_argument('--noise_level', default=0.0, type=float)
parser.add_argument('--linear_transform', default=False, action='store_true')
parser.add_argument('--quantization', default=False, action='store_true')
parser.add_argument('--m', default=32, type=int)

args = parser.parse_args()

index_files = glob.glob(args.passage_reps)
print(f'Pattern match found {len(index_files)} files; loading them into index.')

p_reps_0, p_lookup_0 = pickle_load(index_files[0])
shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
if len(index_files) > 1:
    shards = tqdm(shards, desc='Transform passage embedding shards', total=len(index_files))

q_reps, q_lookup = pickle_load(args.query_reps)
q_reps = transform(q_reps, args.noise_level, args.linear_transform)


if not os.path.exists(args.query_save_to):
    os.makedirs(args.query_save_to)
with open(os.path.join(args.query_save_to, args.query_reps.split('/')[-1]), 'wb') as f:
    pickle.dump((q_reps, q_lookup), f)

all_reps = []
all_lookups = []
if not os.path.exists(args.passage_save_to):
    os.makedirs(args.passage_save_to)
for (p_reps, p_lookup), file in zip(shards, index_files):
    p_reps = transform(p_reps, args.noise_level, args.linear_transform)
    if args.quantization:
        all_reps.append(p_reps)
        all_lookups += p_lookup
    else:
        with open(os.path.join(args.passage_save_to, file.split('/')[-1]), 'wb') as f:
            pickle.dump((p_reps, p_lookup), f)

if args.quantization:
    all_reps = np.concatenate(all_reps, axis=0)
    index = faiss.IndexPQ(all_reps.shape[1], args.m, 8, faiss.METRIC_INNER_PRODUCT)  # nbits=8 always, 256 centroids per sub-vector
    # index.sa_encode(all_reps[:3])
    print('Training PQ index...')
    index.train(all_reps)
    print('Adding docs into PQ index...')
    index.add(all_reps)
    # index.search(all_reps[:3], 5)
    faiss.write_index(index, os.path.join(args.passage_save_to, 'index.faiss'))
    with open(os.path.join(args.passage_save_to, 'lookup.pkl'), 'wb') as f:
        pickle.dump(all_lookups, f)
