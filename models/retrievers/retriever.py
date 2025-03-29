from abc import ABC, abstractmethod

class Retriever(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, kwargs):
        pass

    @abstractmethod
    def collate_fn(self, batch, query_or_doc=None ):
        pass

    @abstractmethod
    def similarity_fn(self, q_embs, doc_embs):
         pass