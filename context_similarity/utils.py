import torch
import numpy as np
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,RecursiveJsonSplitter


def get_model(model,model_kwargs,encode_kwargs):
    if isinstance(model,str):
        if model.lower()=='databricks':
            model = DatabricksEmbeddings(endpoint='databricks-bge-large-en')
        else:
            try:
                print(f'Hugging Face Model:{model}')
                model_name = model # BAAI/bge-large-en-v1.5 or BAAI/bge-m3
                # model_name = "BAAI/bge-large-en-v1.5"
                # model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                # encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
                model = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
            except:
                raise TypeError('Please eneter a valid type: Databricks, Huggingface or model instance')
    else: model = model
    return model

def get_splitter(chunk_size:int,chunk_overlap:int,separators=None):
    splitter = RecursiveCharacterTextSplitter(separators=separators,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter

def embed(model,sentence:str or list):
    if isinstance(sentence,str):
        sentence = sentence.strip()
        embedding = model.embed_query(sentence)
    elif isinstance(sentence,list):
        sentence = [s.strip() for s in sentence]
        embedding = model.embed_documents(sentence)
    else:
        raise TypeError('Only strings or list of strings could be embedded!')
    return embedding

def split(splitter, text:str):
    if isinstance(text,str):
        return splitter.split_text(text)
    if isinstance(text,list):
        return [splitter.split_text(t) for t in text]

def score_matrix(arr1,arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    if hasattr(arr1,'__iter__') and hasattr(arr2,'__iter__'):
        if arr1.shape[1]==arr2.shape[1]:
            pass
        else:
            print('Context Embedding should have the same dimensions!')
            return
    else:
        print('Inputs must be iterable!')
        return

    return np.matmul(arr1, arr2.T)

def context_score(text1,text2,model,splitter):
    mat1 = split(splitter,text1)
    mat2 = split(splitter,text2)
    mat1 = embed(model,mat1)
    mat2 = embed(model,mat2)
    score = score_matrix(mat1,mat2)
    return score

def evaluate(score:np.array):
    precision = np.max(score,axis=0).mean()
    recall = np.max(score,axis=1).mean()
    f1score = 2*(precision*recall/(precision+recall))
    return (precision,recall,f1score)
