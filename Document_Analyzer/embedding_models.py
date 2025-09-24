from typing import List, Dict, Any
import os
from abc import ABC, abstractmethod
import numpy as np
from gensim.models import Word2Vec
from transformers import BertModel, BertTokenizer
import openai
from sklearn.metrics.pairwise import cosine_similarity
import torch
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embeddings for given text"""
        pass

class Word2VecEmbedding(EmbeddingModel):
    def __init__(self, vector_size: int = 128):
        self.model = Word2Vec(vector_size=vector_size, window=5, min_count=1)
        self.vector_size = vector_size
        
    def train(self, sentences: List[List[str]]):
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=10)
    
    def embed(self, text: str) -> np.ndarray:
        # Use BERT tokenizer's max length for consistency
        max_length = 512
        words = text.lower().split()[:max_length]
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not word_vectors:
            return np.zeros(self.vector_size)
        return np.mean(word_vectors, axis=0)

class BertEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = 'prajjwal1/bert-tiny', vector_size: int = 128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.cache = {}
        self.vector_size = vector_size
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        
    def embed(self, text: str) -> np.ndarray:
        # Check cache first
        if text in self.cache:
            return self.cache[text]
            
        # Generate embedding
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
        # Cache the result
        self.cache[text] = embedding
        return embedding

class AdaEmbedding(EmbeddingModel):
    def __init__(self, vector_size: int = 128):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.vector_size = vector_size
        
    def embed(self, text: str) -> np.ndarray:
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            # Ada embeddings are 1536 dimensions, so we'll use dimensionality reduction
            embedding = np.array(response.data[0].embedding)
            # Use average pooling for dimensionality reduction
            pool_size = len(embedding) // self.vector_size
            if pool_size > 1:
                embedding = embedding[:pool_size * self.vector_size].reshape(-1, pool_size).mean(axis=1)
            else:
                # If original size is smaller, use truncation/padding
                if len(embedding) > self.vector_size:
                    embedding = embedding[:self.vector_size]
                elif len(embedding) < self.vector_size:
                    embedding = np.pad(embedding, (0, self.vector_size - len(embedding)))
            return embedding
        except Exception as e:
            print(f"Error getting Ada embedding: {e}")
            return np.zeros(self.vector_size)
