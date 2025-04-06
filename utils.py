'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import datasets
import random 
import json
import shutil
import pytrec_eval
import os 
import torch
import time
import glob
import warnings
import numpy as np
import torch

from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm

def init_experiment(config, experiments_folder, index_folder, runs_folder, run_name, overwrite_exp=False, continue_batch=None):
    experiment_folder = os.path.join(experiments_folder, run_name)
    print(f"- experiment folder: {experiment_folder}")
    
    if os.path.exists(experiment_folder) and overwrite_exp:
        shutil.rmtree(experiment_folder)
    if os.path.exists(experiment_folder) and continue_batch == None:
        raise ValueError(f"Experiment folder {experiment_folder} already exists. Please delete it or specify a different run name.")

    os.makedirs(experiments_folder, exist_ok=True)
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(index_folder, exist_ok=True)
    os.makedirs(runs_folder, exist_ok=True)

    OmegaConf.save(config=config, f=f"{experiment_folder}/config.yaml")

    print("################### Experiment config ###################")
    print(OmegaConf.to_yaml(config))
    print("################### Experiment config ###################")

    return run_name, experiment_folder
    

def get_ranking_filename(runs_folder, query_dataset, doc_dataset, retriever_name, dataset_split, retrieve_top_k, query_generator_name):
    query_gen_add = "" if query_generator_name == "copy" else f".{query_generator_name}"
    return f'{runs_folder}/run.retrieve.top_{retrieve_top_k}.{query_dataset}.{doc_dataset}.{dataset_split}.{retriever_name}{query_gen_add}.trec'


def get_index_path(index_folder, dataset_name, model_name, query_or_doc, dataset_split='', query_generator_name='copy'):
    dataset_split = dataset_split + '_' if dataset_split != '' else ''
    query_gen_add = "" if query_generator_name == "copy" or query_or_doc=="doc" else f".{query_generator_name}"
    return os.path.join(index_folder,f'{dataset_name}_{dataset_split}{query_or_doc}_{model_name}{query_gen_add}')


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


def get_by_id(dataset, ids, field=None):
    # if single id is passed cast it to list
    if not isinstance(ids, list):
        ids = [ids]
    idxs = [ dataset.id2index[id_] for id_ in ids if id_ in dataset.id2index]
    if field != None:
        return dataset[idxs][field] if field in dataset[idxs] else []
    else:
        return idxs


def prepare_dataset_from_ids(dataset, q_ids, d_ids, multi_doc=False, query_field="content"):
    if q_ids is None and d_ids is None:
        dataset_dict = {
            "query": dataset["query"][query_field],
            "q_id": dataset["query"]["id"],
        }
        # label이나 ranking_label이 있으면 추가
        dataset_dict.update({'label': dataset['query']['label'] } if 'label' in dataset['query'].features else {})
        dataset_dict.update({'ranking_label': dataset['query']['ranking_label']} if 'ranking_label' in dataset['query'].features else {})
        return datasets.Dataset.from_dict(dataset_dict)
    else:
        if d_ids: # d_ids가 None이 아니면 확인
            assert isinstance(d_ids[0][0], str), f"Document ID type is not string: {d_ids[0][0]}"
        if 'doc' in dataset and hasattr(dataset['doc'], 'id2index') and dataset['doc'].id2index: # doc 데이터셋과 id2index가 있을 때만 검사
            assert isinstance(list(dataset['doc'].id2index.keys())[0], str), "Dataset doc ID type is not string, retrieval will fail. Convert to string."

        dataset_dict = defaultdict(list) # multi_doc=False 일 때 사용될 수 있음 (현재 코드에서는 직접 사용되지 않음)

        labels = get_by_id(dataset['query'], q_ids, 'label')
        ranking_labels = get_by_id(dataset['query'], q_ids, 'ranking_label')
        queries = get_by_id(dataset['query'], q_ids, query_field)

        def mygen():
            for i, q_id in tqdm(enumerate(q_ids), desc='Fetching data from dataset...', total=len(q_ids)):
                docs = get_by_id(dataset['doc'], d_ids[i], 'content')
                d_ids_ = d_ids[i]
                doc_idxs = get_by_id(dataset['doc'], d_ids[i]) # ID에 해당하는 인덱스 가져오기

                if multi_doc:
                    x = {'doc': docs, 'query': queries[i], 'q_id': q_id, 'd_id': d_ids_, 'd_idx': doc_idxs}
                    if labels and i < len(labels):
                         x['label'] = labels[i]
                    if ranking_labels and i < len(ranking_labels):
                         x['ranking_labels'] = ranking_labels[i]
                    yield x
                else:
                    # 각 (쿼리, 문서) 쌍을 별도의 데이터로 만듬
                    for d_id, doc, d_idx in zip(d_ids_, docs, doc_idxs):
                        x = {'d_id': d_id, 'd_idx': d_idx, 'doc': doc, 'query': queries[i], 'q_id': q_id}
                        if labels and i < len(labels): # labels 리스트 확인
                            x['label'] = labels[i]
                        if ranking_labels and i < len(ranking_labels): # ranking_labels 리스트 확인
                            x['ranking_labels'] = ranking_labels[i]
                        yield x

        return datasets.Dataset.from_generator(mygen)




def left_pad(sequence: torch.LongTensor, max_length: int, pad_value: int) -> torch.LongTensor:
    """
    Helper function to perform left padding
    torch.long
    """
    pad_size = max_length - len(sequence)
    padding = torch.full((pad_size,), pad_value, dtype=torch.long)
    return torch.cat([padding, sequence])


def eval_retrieval_kilt(experiment_folder, qrels_folder, query_dataset_name, doc_dataset_name, split, query_ids, doc_ids, scores, top_k=5, reranking=False, debug=False, write_trec=True):
    #only evaluate if wikipedia ids are in dataset
    # if all(sublist for sublist in doc_ids):
    #     return
    scores = scores.tolist() if torch.is_tensor(scores) else scores
    reranking_str = 're' if reranking else ''
    qrels_file = get_qrel_ranking_filename(qrels_folder, query_dataset_name, split, debug)
    if not os.path.exists(qrels_file): return
    qrel = json.load(open(qrels_file))
    if "doc_dataset_name" in qrel:
        if qrel["doc_dataset_name"] != doc_dataset_name: return
        qrel.pop("doc_dataset_name")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_1', f'recall_{top_k}'})
    run = defaultdict(dict)
    for i, q_id in enumerate(query_ids):
        for i, (doc_id, score) in enumerate(zip(doc_ids[i], scores[i])):
            # if we have duplicate doc ids (because different passage can map to same wiki page) only write the max scoring passage
            if doc_id not in run[q_id]:
                run[q_id].update({doc_id: float(score)})
            # if there is a higher scoring passage from the same wiki_doc, update the score (maxP)
            elif score >= run[q_id][doc_id]:
                run[q_id].update({doc_id: float(score)})

    if write_trec:
        with open(f'{experiment_folder}/eval_{split}_{reranking_str}ranking_run.trec', 'w') as trec_out:
            for q_id, scores_dict in run.items():
                # Sort the dictionary by scores in decreasing order
                sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
                for i, (doc_id, score) in enumerate(sorted_scores.items()):
                    trec_out.write(f'{q_id}\tQO\t{doc_id}\t{i+1}\t{score}\trun\n')

    metrics_out = evaluator.evaluate(run)
    p_1 = sum([d["P_1"] for d in metrics_out.values()]) / max(1, len(metrics_out))
    recall = sum([d[f"recall_{top_k}"] for d in metrics_out.values()]) / max(1, len(metrics_out))
    
    mean_metrics = {'P_1':p_1, f'recall_{top_k}': recall}
    fname = f"eval_{split}_{reranking_str}ranking_metrics.json"
    write_dict(experiment_folder,  fname, mean_metrics)

def get_qrel_ranking_filename(qrels_folder, dataset_name, split, debug=False):
    dataset_name = dataset_name.replace('_debug', '') if debug else dataset_name
    return f'{qrels_folder}/qrel.{dataset_name}.{split}.json'


def write_generated(out_folder, out_filename, query_ids, questions, instructions, responses, labels, ranking_labels):
    jsonl_list = list()
    for i, (q_id, question, response, instruction, label, ranking_label) in enumerate(zip(query_ids, questions, responses, instructions, labels, ranking_labels)):
        jsonl = {}
        jsonl['q_id'] = q_id
        jsonl['response'] = response
        jsonl['instruction'] = instruction
        jsonl['label'] = label
        jsonl['question'] = question
        jsonl['ranking_label'] = ranking_label
        jsonl_list.append(jsonl)
    write_dict(out_folder, out_filename, jsonl_list)

def write_dict(out_folder, out_filename, dict_to_write):
    with open(f'{out_folder}/{out_filename}', 'w') as fp:
        json.dump(dict_to_write, fp, indent=2)


def print_generate_out(queries, instructions, responses, query_ids, labels, ranking_labels, n=5):
    rand = random.sample(range(len(query_ids)), n)
    for i in rand:
        print('_'*50)
        print('Query ID:', query_ids[i])
        print('Query:', queries[i])
        print('_'*50)
        if instructions[i] != None:
            print('Instruction to Generator:')
            print(instructions[i])
        print()
        print('LLM Answer:')
        print(responses[i])
        print('Label(s):')
        print(labels[i])
        if ranking_labels[i] != None:
            print('Ranking Label(s):')
            print(ranking_labels[i])
        print()
        print()

def get_context_processing_filename(context_processing_folder, query_dataset, doc_dataset, dataset_split, retriever_name, retrieve_top_k, reranker_name, rerank_top_k, generation_top_k, query_generator_name, context_processor_name):
    query_gen_add = "" if query_generator_name == "copy" else f".{query_generator_name}"
    rerank_name = f"rerank.top_{rerank_top_k}.{reranker_name}" if reranker_name is not None else "no_rerank"
    return f'{context_processing_folder}/processed_contexts.{context_processor_name}.retriever.top_{retrieve_top_k}.{retriever_name}.{rerank_name}.generate_top_{generation_top_k}.{query_dataset}.{doc_dataset}.{dataset_split}{query_gen_add}.json'


def get_query_generation_filename(query_generation_folder, query_dataset, query_generator_name, split):
    return f'{query_generation_folder}/generated_queries.{query_dataset}.{split}.{query_generator_name}.json'


def format_time(field_name, generation_time):
    return {field_name: time.strftime("%H:%M:%S.{}".format(str(generation_time % 1)[2:])[:11], time.gmtime(generation_time))}