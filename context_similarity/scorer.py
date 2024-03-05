import torch
from utils import (get_model,get_splitter,context_score,evaluate)


class Scorer:
    def __init__(self,
                 model_type = None,
                 model_kwargs:dict=None,
                 encode_kwargs=None,
                 chunk_size:int = None,
                 chunk_overlap:int = None,
                 separators:list=None):
        assert (model_type is not None and chunk_size is not None and chunk_overlap is not None,
                'model, chunk size and overlap should be specified.')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators=separators
        self.model_type = model_type
        self.model_kwargs =  {'device': 'cuda' if torch.cuda.is_available() else 'cpu'} if model_kwargs is None else model_kwargs
        self.encode_kwargs = {'normalize_embeddings': True} if encode_kwargs is None else encode_kwargs
        self.model = get_model(model_type,self.model_kwargs,self.encode_kwargs)
        self.splitter = get_splitter(chunk_size,chunk_overlap,separators)


    def update(self):
        self.model = get_model(self.model_type, self.model_kwargs, self.encode_kwargs)
        self.splitter = get_splitter(self.chunk_size, self.chunk_overlap, self.separators)

    def score(self,text,target):
        scores = context_score(text,target,self.model,self.splitter)
        return scores, evaluate(scores)

