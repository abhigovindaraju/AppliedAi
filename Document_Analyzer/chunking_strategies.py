from typing import List
from abc import ABC, abstractmethod
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from transformers import pipeline

class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks according to strategy"""
        pass

class FixedSizeChunking(ChunkingStrategy):
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into fixed-size chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

class SemanticChunking(ChunkingStrategy):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into semantic chunks based on sentence and paragraph boundaries"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_size = len(sent_text.split())
            
            if current_size + sent_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent_text]
                current_size = sent_size
            else:
                current_chunk.append(sent_text)
                current_size += sent_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class RecursiveChunking(ChunkingStrategy):
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text recursively with summarization for large chunks"""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        # Initial split into large chunks
        for i in range(0, len(words), chunk_size * 2):
            large_chunk = ' '.join(words[i:i + chunk_size * 2])
            
            if len(large_chunk.split()) > chunk_size:
                # Summarize large chunks
                summary = self.summarizer(large_chunk, max_length=chunk_size, 
                                       min_length=chunk_size//2, do_sample=False)[0]['summary_text']
                chunks.append(summary)
            else:
                chunks.append(large_chunk)
        
        return chunks

class AdaptiveChunking(ChunkingStrategy):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text adaptively based on content complexity and semantic boundaries"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_complexity = self._calculate_complexity(sent_text)
            sent_size = len(sent_text.split())
            
            # Adjust chunk size based on complexity
            adjusted_size = chunk_size * (1 - sent_complexity)
            
            if current_size + sent_size > adjusted_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent_text]
                current_size = sent_size
            else:
                current_chunk.append(sent_text)
                current_size += sent_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1) based on various factors"""
        doc = self.nlp(text)
        
        # Factors to consider for complexity
        avg_word_length = sum(len(token.text) for token in doc) / len(doc)
        named_entities = len(doc.ents)
        sentence_length = len(doc)
        
        # Normalize and combine factors
        complexity = (
            (avg_word_length / 10) * 0.4 +  # Max assumed avg word length is 10
            (named_entities / 10) * 0.3 +    # Max assumed NEs per sentence is 10
            (sentence_length / 50) * 0.3     # Max assumed sentence length is 50
        )
        
        return min(max(complexity, 0), 1)  # Ensure result is between 0 and 1
