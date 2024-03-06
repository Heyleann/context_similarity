import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import (get_model,get_splitter,context_score,evaluate,split,embed)


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
        self.scores = None


    def update(self):
        self.model = get_model(self.model_type, self.model_kwargs, self.encode_kwargs)
        self.splitter = get_splitter(self.chunk_size, self.chunk_overlap, self.separators)

    def score(self,text,target):
        self.scores = context_score(text,target,self.model,self.splitter)
        return self.scores

    def evaluate(self):
        return evaluate(self.scores)

    def display_chunks(self,text,target):
        text = split(self.splitter,text)
        target = split(self.splitter,target)
        return text, target

    def display_embeddings(self,text,target):
        text = embed(self.model,split(self.splitter, text))
        target = embed(self.model,split(self.splitter, target))
        return text, target

    def plot(self,text,target,title):
        scores = self.score(text,target)
        mask = np.zeros_like(scores)
        mask[np.triu_indices_from(mask)]=True
        with sns.axes_style("white"):
            ax = sns.heatmap(scores,mask=mask,square=True,cmap = "crest",annot=True,fmt=".2f")
            ax.set(xlabel="Target",ylabel='Text',title=f'{title}')
            plt.show()
