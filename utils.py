import os
import glob
import torch
from tqdm import tqdm
from collections import defaultdict

def get_ranking_filename(runs_folder, query_dataset, doc_dataset, retriever_name, dataset_split, retrieve_top_k):
    return f'{runs_folder}/run.retrieve.top_{retrieve_top_k}.{query_dataset}.{doc_dataset}.{dataset_split}.{retriever_name}.trec'

def get_index_path(index_folder, dataset_name, model_name, query_or_doc, dataset_split=''):
    dataset_split = dataset_split + '_' if dataset_split != '' else ''
    return os.path.join(index_folder,f'{dataset_name}_{dataset_split}{query_or_doc}_{model_name}')

def write_trec(fname, q_ids, d_ids, scores):
    with open(fname, 'w') as fout:
        for i, q_id in enumerate(q_ids):
            for rank, (d_id, score) in enumerate(zip(d_ids[i], scores[i])):
                fout.write(f'{q_id}\tq0\t{d_id}\t{rank+1}\t{score}\trun\n')

def load_trec(fname):
    # read file
    trec_dict = defaultdict(list)
    for l in tqdm(open(fname), desc=f'Loading existing trec run {fname}'):
        q_id, _, d_id, _, score, _ = l.split('\t')
        trec_dict[q_id].append((d_id, score))
    q_ids, d_ids, scores = list(), list(), list()
    for q_id in trec_dict:
        q_ids.append(q_id)
        d_ids_q, scores_q = list(), list()
        for d_id, score in trec_dict[q_id]:
            d_ids_q.append(d_id)
            scores_q.append(float(score))
        d_ids.append(d_ids_q)
        scores.append(scores_q)
    return q_ids, d_ids, scores

def load_embeddings(index_path):
    try:
        emb_files = glob.glob(f'{index_path}/*.pt')
        sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        embeds = list()
        for i, emb_file in enumerate(tqdm(sorted_emb_files, total=len(sorted_emb_files), desc=f'Load embeddings...')):
            emb_chunk = torch.load(emb_file)
            embeds.append(emb_chunk)
        embeds = torch.concat(embeds)
    except RuntimeError:
        # RuntimeError: torch.cat(): expected a non-empty list of Tensors 
        # --> embeddings were not found
        raise RuntimeError("No embeddings found. Check .trec run file name if you are running oracle provenance.")
    except Exception as e:
        print("Exception occured: ", e)
        raise IOError(f'Embedding index corrupt. Please delete folder "{index_path}" and run again.')
    return embeds