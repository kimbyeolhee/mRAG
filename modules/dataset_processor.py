import datasets
from datasets import Dataset
import os
from collections import defaultdict
import csv
from tqdm import tqdm
import pickle
from hydra.utils import instantiate
import json
from functools import partial
import random
from typing import Dict

# Base class that every processor interhits from 
class Processor(object):
    """
    Base dataset processor class.
    """
    
    def __init__(self, 
        dataset_name: str,
        split: str, 
        out_folder: str, 
        num_proc: int, 
        overwrite: bool, 
        debug: bool, 
        shuffle_labels: bool
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.num_proc = num_proc
        self.out_folder = out_folder
        self.overwrite = overwrite
        self.debug = debug
        self.shuffle_labels = shuffle_labels

    def process() -> Dataset:
        raise NotImplementedError()
    
    def add_index(self, dataset: Dataset) -> Dataset:
        dataset = dataset.add_column("index", range(len(dataset)))    
        return dataset
    
    def get_index_to_id(self, dataset: Dataset) -> Dict[str, int]:
        if 'index' not in dataset.features:
            dataset = self.add_index(dataset)
        return dict(zip(dataset["id"], dataset["index"]))
    
    def shuffled_labels_as_content(self, dataset: Dataset) -> Dataset:
        random.seed(42)
        col = dataset['label']
        random.shuffle(col)
        dataset_dict = dataset.to_dict()
        dataset_dict['ranking_label'] = [el[0] for el in col]
        return datasets.Dataset.from_dict(dataset_dict)

    def get_dataset(self) -> Dataset:
        print(f"Processing dataset {self.dataset_name} in {self.split} split ")
        debug_str = '_debug' if self.debug else ''
        assert self.dataset_name != None # dataset name needs to be set in processor class
        out_folder = os.path.join(f'{self.out_folder}', f'{self.dataset_name}_{self.split}')
        if os.path.exists(out_folder) and not self.overwrite:
            dataset = datasets.load_from_disk(out_folder)
            if self.debug:
                dataset = dataset.select(range(min(len(dataset), 50)))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
            #id2index = self.tsv_to_dict(f'{out_folder}/id2index.csv')
            id2index = pickle.load(open(f'{out_folder}/id2index.p', 'rb'))
            dataset.id2index = id2index
        else:
            dataset = self.process()
            dataset.save_to_disk(out_folder)
            id2index = self.get_index_to_id(dataset)
            pickle.dump(id2index, open(f'{out_folder}/id2index.p', 'wb'))
            if self.debug:
                dataset = dataset.select(range(min(len(dataset), 50)))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
            dataset.id2index = id2index
            #self.dict_to_tsv(id2index, f'{out_folder}/id2index.csv')
        dataset.name = self.dataset_name + debug_str
        return dataset
    
    def dict_to_tsv(self, id_to_index: Dict[str, int], file_path: str) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                # Write data
                for id, index in id_to_index.items():
                    row = f"{id}\t{index}\n"
                    file.write(row)
        except Exception as e:
            print(f"Error writing id2index file: {e}")

    def tsv_to_dict(self, file_path: str) -> Dict[str, int]:
        try:
            id_to_index = {}
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    if len(row) == 2:
                        id, index = row
                        id_to_index[id] = int(index)
            return id_to_index
        except Exception as e:
            print(f"Error loading id2index file: {e}")
            return None



class Wiki_monolingual_100w(Processor):

    def __init__(self, lang, *args, **kwargs):
        dataset_name = 'wiki-100w-' + lang
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang

    def process(self):
        hf_name = 'wikimedia/wikipedia'
        subset = "20231101." + self.lang
        dataset = datasets.load_dataset(hf_name, subset, num_proc=self.num_proc)[self.split]

        def map_100w(sample, num_words=100):
            wiki_id = sample['id']
            title = sample['title']
            doc = sample["text"]
            if self.lang not in ["zh", "ja", "th"]:
                words = doc.split()
            else:
                words = list(doc)
            paragraphs = [title + '. ' + " ".join(words[i:i + num_words]) for i in range(0, len(words), num_words)]
            wiki_ids = [wiki_id] * len(paragraphs)
            return {'paragraphs': paragraphs, "wiki_ids": wiki_ids}
        
        kilt_dataset = dataset.map(map_100w, num_proc=self.num_proc)
        paragraphs = [el for sublist in kilt_dataset['paragraphs'] for el in sublist]
        wiki_ids = [el for sublist in kilt_dataset['wiki_ids'] for el in sublist]
        dataset = datasets.Dataset.from_dict({'content': paragraphs, 'wikipedia_id': wiki_ids})
        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        del kilt_dataset
        return dataset


class ProcessDatasets:
    """
    Static class to process datasets.

    Methods:
    - process: Process the datasets.
    - check_instantiate: Check the instantiation of datasets.
    """
    
    @staticmethod
    def process(datasets, out_folder='datasets', num_proc=1, overwrite=False, debug=False, shuffle_labels=False):
        def sanity_checks(dataset):
            for example in tqdm(dataset, 'Checking dataset..'):
                for field_name, field_value in example.items():
                    if field_value is None:
                        raise ValueError(f"Found None value in '{field_name}' field.")
                    elif isinstance(field_value, list) and None in field_value:
                        raise ValueError(f"Found None in list in '{field_name}' field.")
                    elif isinstance(field_value, str) and len(field_value.strip()) == 0:
                        raise ValueError(f"Found empty value in '{field_name}' field.")
                    elif isinstance(field_value, list) and len(field_value) == 0:
                        raise ValueError(f"Found empty list in '{field_name}' field.")
                

        processed_datasets = defaultdict(dict)
        for split in datasets:
            for query_or_doc in datasets[split]:
                if datasets[split][query_or_doc] != None:
                    processor_init_args = datasets[split][query_or_doc]['init_args']
                    processor = instantiate(
                        processor_init_args, 
                        out_folder=out_folder, 
                        num_proc=num_proc, 
                        overwrite=overwrite, 
                        debug= debug if query_or_doc == 'query' else False, 
                        shuffle_labels= shuffle_labels if query_or_doc == 'query' else False
                        )
                    dataset = processor.get_dataset()
                    if query_or_doc == 'query':
                        sanity_checks(dataset)
                    processed_datasets[split][query_or_doc] = dataset
                else:
                    processed_datasets[split][query_or_doc] = None
        return processed_datasets
    
    @staticmethod
    def check_instantiate(datasets, out_folder='datasets', num_proc=1, overwrite=False, debug=False):
        processed_datasets = defaultdict(dict)
        for split in datasets:
            for query_or_doc in datasets[split]:
                if datasets[split][query_or_doc] != None:
                    processor_init_args = datasets[split][query_or_doc]['init_args']
                    processor = instantiate(
                        processor_init_args, 
                        out_folder=out_folder, 
                        num_proc=num_proc, 
                        overwrite=overwrite, 
                        debug= debug if query_or_doc == 'query' else False, 
                        oracle_provenance=  False, 
                        shuffle_labels= False
                        )          
        return True


class MKQA(Processor):
    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'mkqa_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        mkqa = datasets.load_dataset('mkqa', trust_remote_code=True)
        kilt_nq = datasets.load_dataset("kilt_tasks", "nq")

        mkqa_ids = {s['example_id']:i for i, s in enumerate(mkqa[self.split])}
        kilt_nq_train_ids = {s['id']:i for i, s in enumerate(kilt_nq[self.split])}

        overlap_ids = set(mkqa_ids.keys()).intersection(set(kilt_nq_train_ids.keys()))
        overlap_mkqa = mkqa['train'].select([mkqa_ids[i] for i in overlap_ids])
        overlap_kilt_nq = kilt_nq['train'].select([kilt_nq_train_ids[i] for i in overlap_ids])        
        dataset = overlap_kilt_nq.add_column(f"content", [sample['queries'][self.lang] for sample in overlap_mkqa])    
        # discarding empty answers
        dataset = dataset.add_column(f"label", [[a['text'] for a in sample['answers'][self.lang] if not a['text']==None] for sample in overlap_mkqa])
        # filter out samples with empty answer
        dataset = dataset.filter(lambda example: len(example['label'])>0)

        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el[f'answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})        
        dataset = dataset.remove_columns(['meta'])
        return dataset