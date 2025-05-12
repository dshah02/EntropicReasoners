from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch
import os
from pathlib import Path
from npeet import entropy_estimators as ee
from sklearn.decomposition import PCA

_CACHED_MODEL = None #is this how you do it?

def get_cached_model(model_name: str = '/scratch/gpfs/oy3975/cache/Linq-Embed-Mistral') -> SentenceTransformer:
    #Avoids repeatedly redownloading
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = SentenceTransformer(model_name, trust_remote_code=True, local_files_only=True)
    return _CACHED_MODEL

def compute_embedding_label_mi(texts: List[str], 
                             labels: List[int], 
                             model_name: str = '/scratch/gpfs/oy3975/cache/multilingual',
                             n_shuffles: int = 10,
                             compute_control: bool = True) -> dict:
    labels = np.array(labels)

    model = get_cached_model(model_name)
    embeddings = model.encode(texts)

    total_mi = ee.mi(embeddings, labels) #aggregate entropic estimators?, whatever this means

    result = {"total_mi": float(total_mi)}
    if compute_control:
        total_mi_control = ee.shuffle_test(ee.mi, embeddings, labels, ci=0.95, ns=n_shuffles)
        result["total_mi_control"] = float(total_mi_control[0])
    
    return result
