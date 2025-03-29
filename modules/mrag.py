import os
import shutil
from datasets import Dataset

from modules.retrieve import Retrieve
from modules.dataset_processor import ProcessDatasets

from utils import (
    get_ranking_filename,
    write_trec, load_trec,
    get_index_path
)

class mRAG:
    def __init__(self, 
                retriever=None,
                generator=None,
                dataset=None,
                runs_folder=None,
                run_name=None,
                processing_num_proc=1,
                dataset_folder='datasets/',
                index_folder='indexes/',
                generated_query_folder='generated_queries/',
                experiments_folder='experiments/',
                overwrite_datasets=False,
                overwrite_exp=False,
                overwrite_index=False,
                pyserini_num_threads=1,
                retrieve_top_k=1,
                config=None,
                debug=False):

        
        retriever_config = retriever
        generator_config = generator
        dataset_config = dataset

        self.debug = debug
        self.dataset_folder = dataset_folder
        self.runs_folder = runs_folder
        if self.runs_folder:
            os.makedirs(self.runs_folder, exist_ok=True)
        self.generated_query_folder = generated_query_folder
        if self.generated_query_folder:
            os.makedirs(self.generated_query_folder, exist_ok=True)
        self.experiments_folder = experiments_folder
        if self.experiments_folder:
            os.makedirs(self.experiments_folder, exist_ok=True)
        self.pyserini_num_threads = pyserini_num_threads
        self.overwrite_exp = overwrite_exp
        self.overwrite_index = overwrite_index
        self.run_name = run_name
        self.index_folder = index_folder
        if self.index_folder:
            os.makedirs(self.index_folder, exist_ok=True)
        self.config = config
        self.retrieve_top_k = retrieve_top_k


        # process dataset
        self.datasets = ProcessDatasets.process(
            dataset_config, 
            out_folder=self.dataset_folder, 
            num_proc=processing_num_proc,
            overwrite=overwrite_datasets,
            debug=debug,
            shuffle_labels=True if generator_config is not None and generator_config.init_args.model_name == 'random_answer' else False,
            )

        # init retriever
        self.retriever = Retrieve(
                    **retriever_config,
                    pyserini_num_threads=self.pyserini_num_threads,
                    ) if retriever_config is not None else None


    def eval(self, dataset_split):
        print("mRAG eval")

        dataset = self.datasets[dataset_split]

        query_dataset_name = self.datasets[dataset_split]['query'].name
        doc_dataset_name = self.datasets[dataset_split]['doc'].name if "doc" in self.datasets[dataset_split] else None

        dataset['query'] = dataset['query'].add_column("generated_query", dataset['query']["content"])

        print(f"Query dataset: {query_dataset_name}")
        print(f"Doc dataset: {doc_dataset_name}")

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

        # show example query and retrieved docs
        print(f"Example query: {dataset['query'][0]['content']}")
        print(f"Retrieved docs: {doc_ids[0]}")
        # retrieved docs 전체 text 출력
        print(f"Retrieved docs: {dataset['doc'][doc_ids[0]]['content']}")
        print(f"Retrieved scores: {scores[0]}")
    

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
                )
        doc_embeds_path = get_index_path(self.index_folder, doc_dataset_name, self.retriever.get_clean_model_name(), 'doc')
        query_embeds_path = get_index_path(self.index_folder, query_dataset_name, self.retriever.get_clean_model_name(), 'query', dataset_split=dataset_split)

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

        return query_ids, doc_ids, scores