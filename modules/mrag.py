import os
import shutil
import time
import json
from tqdm import tqdm
from hydra.utils import instantiate
import pandas as pd
import numpy as np

from utils import (
    eval_retrieval_kilt, init_experiment, move_finished_experiment,
    write_trec, prepare_dataset_from_ids, load_trec,
    print_generate_out,
    write_generated, write_dict, get_by_id, get_index_path, get_query_generation_filename,
    get_context_processing_filename,
    get_ranking_filename, format_time
)

from modules.retrieve import Retrieve
from modules.dataset_processor import ProcessDatasets
from modules.generate_query import GenerateQueries
from modules.process_context import ProcessContext
from modules.metrics import RAGMetrics



class mRAG:
    def __init__(self, 
                retriever=None,
                generator=None,
                query_generator=None, # 나중에 query 번역이 필요할지도 몰라서 선언
                context_processor=None, # 나중에 context 번역이 필요할지도 몰라서 선언
                runs_folder=None,
                run_name=None,
                dataset=None,
                processing_num_proc=1,
                dataset_folder='datasets/',
                index_folder='indexes/',
                generated_query_folder='generated_queries/',
                processed_context_folder='processed_contexts/',
                experiments_folder='experiments/',
                qrels_folder='qrels/',
                overwrite_datasets=False,
                overwrite_exp=False,
                overwrite_index=False,
                retrieve_top_k=1,
                generation_top_k=1,
                pyserini_num_threads=1,
                config=None,
                debug=False,
                continue_batch=None,
                prompt=None,
                **kwargs
                ):

        retriever_config = retriever
        generator_config = generator
        query_generator_config = query_generator
        context_processor_config = context_processor
        dataset_config = dataset

        if context_processor_config is None:
                    context_processor_config = config.context_processor if hasattr(config, 'context_processor') else None

        if query_generator_config is None:
            query_generator_config = {"init_args": {"_target_": "models.query_generators.copy.CopyQuery"}}
                

        self.debug = debug
        self.dataset_folder = dataset_folder
        self.experiments_folder = experiments_folder
        self.runs_folder = runs_folder
        self.generated_query_folder = generated_query_folder
        self.qrels_folder = qrels_folder # 랭킹 평가 파일 저장 폴더
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder
        self.config = config
        self.retrieve_top_k = retrieve_top_k
        self.generation_top_k = generation_top_k
        self.pyserini_num_threads = pyserini_num_threads # bm25 사용안하면 지우기
        self.overwrite_exp = overwrite_exp
        self.overwrite_index = overwrite_index


        # init experiment
        self.run_name, self.experiment_folder = init_experiment(config, experiments_folder, index_folder, runs_folder, run_name, overwrite_exp=self.overwrite_exp, continue_batch=continue_batch)

        # process dataset
        self.datasets = ProcessDatasets.process(
            dataset_config, 
            out_folder=self.dataset_folder, 
            num_proc=processing_num_proc,
            overwrite=overwrite_datasets,
            debug=debug,
            shuffle_labels=True if generator_config is not None and generator_config.init_args.model_name == 'random_answer' else False,
            )

        self.metrics = {
            "train": RAGMetrics,
            # lookup metric with dataset name (tuple: dataset_name, split) 
            "dev": RAGMetrics, 
            "test": None,
        }

        # init retriever
        self.retriever = Retrieve(
                    **retriever_config,
                    pyserini_num_threads=self.pyserini_num_threads,
                    continue_batch=continue_batch,
                    ) if retriever_config is not None else None

        # init generator
        self.generator = instantiate(generator_config.init_args, prompt=prompt) if generator_config is not None else None

        self.query_generator = GenerateQueries(self.generator, **query_generator_config) if query_generator_config is not None else None

        self.context_processor = ProcessContext(**context_processor_config) if context_processor_config is not None else None
        


    def eval(self, dataset_split):
        print("################### mRAG eval start ###################")
        
        dataset = self.datasets[dataset_split]        
        query_dataset_name = self.datasets[dataset_split]['query'].name
        doc_dataset_name = self.datasets[dataset_split]['doc'].name if "doc" in self.datasets[dataset_split] else None

        # query generation (baseline은 query_generator="copy")
        if self.retriever is not None:
            dataset = self.generate_query(
                dataset,
                query_dataset_name, 
                dataset_split, 
            )    
        
        # retrieve
        if self.retriever is not None:
            query_ids, doc_ids, scores = self.retrieve(
                dataset,
                query_dataset_name,
                doc_dataset_name,
                dataset_split,
                self.retrieve_top_k
            )
        else:
            query_ids, doc_ids = None, None     

        doc_ids = [doc_ids_q[:self.generation_top_k] for doc_ids_q in doc_ids] if doc_ids != None else doc_ids 

        gen_dataset = prepare_dataset_from_ids(
            dataset, 
            query_ids, 
            doc_ids,
            multi_doc=True,
            query_field="content",
            )
        
        # process context
        if self.context_processor is not None and self.retriever is not None:
            gen_dataset = self.process_context(
                                               gen_dataset, 
                                               query_dataset_name, 
                                               doc_dataset_name, 
                                               dataset_split
                                              )

        # generate
        if self.generator is not None:
            questions, _, predictions, references = self.generate(
                gen_dataset, 
                dataset_split, 
                )
            # eval metrics
            self.eval_metrics(
                dataset_split, 
                questions, 
                predictions, 
                references
                )

        move_finished_experiment(self.experiment_folder)        


    def generate_query(self, dataset, query_dataset_name, dataset_split):
        id2index = dataset['query'].id2index
        if self.query_generator.get_clean_model_name() == "copy":
            dataset['query'] = dataset['query'].add_column("generated_query", dataset['query']["content"])
        else:
            gen_query_file = get_query_generation_filename(
                self.generated_query_folder, 
                query_dataset_name, 
                self.query_generator.get_clean_model_name(), 
                dataset_split
            )
            if not os.path.exists(gen_query_file) or self.overwrite_exp or self.overwrite_index:
                print("Generating search queries...")
                generated_queries = self.query_generator.eval(dataset['query'])
                os.makedirs(self.generated_query_folder, exist_ok=True)
                with open(gen_query_file, 'w', encoding='utf-8') as fp: 
                    json.dump({"generated_queries": generated_queries}, fp)
            else:
                print("Using pre-generated search queries...")
                with open(gen_query_file, 'r', encoding='utf-8') as fp: 
                    generated_queries = json.load(fp)["generated_queries"]
            dataset['query'] = dataset['query'].add_column("generated_query", generated_queries)
            shutil.copyfile(gen_query_file, f'{self.experiment_folder}/{gen_query_file.split("/")[-1]}')
        dataset['query'].id2index = id2index
        return dataset


    def retrieve(self, 
                 dataset, 
                 query_dataset_name, 
                 doc_dataset_name,
                 dataset_split, 
                 retrieve_top_k,
                 eval_ranking=True,
                 ):

        ranking_file = get_ranking_filename(
                    self.runs_folder,
                    query_dataset_name,
                    doc_dataset_name,
                    self.retriever.get_clean_model_name(),
                    dataset_split, 
                    retrieve_top_k,
                    self.query_generator.get_clean_model_name()
                )
        doc_embeds_path = get_index_path(self.index_folder, doc_dataset_name, self.retriever.get_clean_model_name(), 'doc')
        query_embeds_path = get_index_path(self.index_folder, query_dataset_name, self.retriever.get_clean_model_name(), 'query', dataset_split=dataset_split, query_generator_name=self.query_generator.get_clean_model_name())

        if not os.path.exists(ranking_file) or self.overwrite_exp or self.overwrite_index:
            print(f"{ranking_file} does not exist, running retrieval")
            out_ranking = self.retriever.retrieve(
                dataset,
                query_embeds_path,
                doc_embeds_path,
                retrieve_top_k,
                overwrite_index=self.overwrite_index
            )
            query_ids, doc_ids, scores = out_ranking["q_id"], out_ranking["doc_id"], out_ranking["score"]
            write_trec(ranking_file, query_ids, doc_ids, scores)

        else:
            print(f"{ranking_file} already exists, skipping retrieval")
            query_ids, doc_ids, scores = load_trec(ranking_file)

        shutil.copyfile(ranking_file, f'{self.experiments_folder}/{ranking_file.split("/")[-1]}')
        if eval_ranking:
            if 'ranking_label' in self.datasets[dataset_split]['query'].features:
                print('Evaluating retrieval...')
                wiki_doc_ids = [get_by_id(self.datasets[dataset_split]['doc'], doc_ids_q, 'wikipedia_id') for doc_ids_q in tqdm(doc_ids, desc='Getting wiki ids...')]
                eval_retrieval_kilt(
                    self.experiment_folder, 
                    self.qrels_folder, 
                    query_dataset_name, 
                    doc_dataset_name,
                    dataset_split, query_ids, 
                    wiki_doc_ids, scores, 
                    top_k=self.generation_top_k, 
                    debug=self.debug,
                    )
        return query_ids, doc_ids, scores


    def process_context(self, gen_dataset, 
                       query_dataset_name, 
                       doc_dataset_name, 
                       dataset_split):
        process_context_file = get_context_processing_filename(
            self.processed_context_folder, 
            query_dataset_name,
            doc_dataset_name,
            dataset_split,
            self.retriever.get_clean_model_name(),
            self.retrieve_top_k,
            self.reranker.get_clean_model_name() if self.reranker is not None else None,
            self.rerank_top_k,
            self.generation_top_k,
            self.query_generator.get_clean_model_name(),
            self.context_processor.get_clean_model_name(),
        )
        if not os.path.exists(process_context_file) or self.overwrite_exp or self.overwrite_index:
            processed_contexts, context_metrics = self.context_processor.eval(gen_dataset['doc'], 
                                                                              gen_dataset['query'])
            os.makedirs(self.processed_context_folder, exist_ok=True)
            with open(process_context_file, 'w', encoding='utf-8') as fp: 
                json.dump({"processed_contexts": processed_contexts,
                           "context_metrics": context_metrics,
                           "original_contexts": gen_dataset['doc'],
                           "queries": gen_dataset['query']}, 
                          fp)
        else:
            with open(process_context_file, 'r', encoding='utf-8') as fp: 
                save = json.load(fp)
                processed_contexts = save["processed_contexts"]
                context_metrics = save["context_metrics"]
        gen_dataset = gen_dataset.remove_columns('doc')
        gen_dataset = gen_dataset.add_column('doc', processed_contexts)
        shutil.copyfile(process_context_file, f'{self.experiment_folder}/{process_context_file.split("/")[-1]}')
        with open(f'{self.experiment_folder}/eval_{dataset_split}_context_metrics.json', 'w', encoding='utf-8') as fout:
            json.dump(context_metrics, fout)
        return gen_dataset


    def generate(self, gen_dataset, dataset_split):
        generation_start = time.time()
        query_ids, questions, instructions, predictions, references, ranking_labels = self.generator.eval(gen_dataset)
        generation_time = time.time() - generation_start

        write_generated(
            self.experiment_folder,
            f"eval_{dataset_split}_out.json",
            query_ids, 
            questions,
            instructions, 
            predictions, 
            references, 
            ranking_labels
        )

        print_generate_out(
            questions,
            instructions,
            predictions,
            query_ids, 
            references,
            ranking_labels,
            )

        if hasattr(self.generator,"total_cost"):
            print(self.generator.total_cost,self.generator.prompt_cost, self.generator.completion_cost)
            write_dict(self.experiment_folder, f"eval_{dataset_split}_generation_cost.json", 
                       {'total_cost':self.generator.total_cost,
                        'prompt_cost':self.generator.prompt_cost,
                        'completion_cost':self.generator.completion_cost}
                        )


        formated_time_dict = format_time("Generation time", generation_time)
        write_dict(self.experiment_folder, f"eval_{dataset_split}_generation_time.json", formated_time_dict)

        return questions, instructions, predictions, references


    def eval_metrics(self, dataset_split, questions, predictions, references):
        if predictions is None and references is None and questions is None:
            return
        out_file = f"{self.experiment_folder}/eval_{dataset_split}_out.json"
        with open(out_file, 'r', encoding='utf-8') as fd:
            generated = json.load(fd)
        generated = pd.DataFrame(generated)
        metrics_out = self.metrics[dataset_split].compute(
        predictions=predictions, 
        references=references, 
        questions=questions
        )
        for m in metrics_out:
            generated[m] = metrics_out[m]
        avg_metrics = {v: np.mean(metrics_out[v]) for v in metrics_out}
        write_dict(self.experiment_folder, f"eval_{dataset_split}_metrics.json", avg_metrics)        
        generated.to_json(out_file, orient='records')
        