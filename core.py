import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from uuid import uuid4
from tqdm import tqdm
from pathlib import Path

# Transformers & Embeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain Components
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Milvus
from langchain_milvus import Milvus, BM25BuiltInFunction

# Milvus
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility



class Embedding:
    def __init__(self, tokenizer, model):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model = model

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embedding_documents(self, input_texts):
        """
        input_texts: list các string chunk sau khi được split nhờ Text Splitter của langchain
        Tạo các vector embedding cho từng chunk này (encode vector)
        """
        embeddings_list = []
        
        for text in tqdm(input_texts, desc="Embedding Documents", ncols=100):
            batch_dict = self.tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            
            embedding = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)        
            
            embeddings_list.append(embedding.squeeze(0).tolist())
            break

        print(f"Generated {len(embeddings_list)} embeddings.")
        return embeddings_list

    def embed_text(self, text):
        """
        Convert input text into an embedding vector.
        """
        batch_dict = self.tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        
        embedding = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)        

        return embedding.squeeze(0)
    

class TextSplitter:
    def __init__(self, separator = "\n", chunk_size = 1000, chunk_overlap = 200, _prefix = True):
        print(f"Loading text splitter")
        self._prefix = _prefix
        self.text_splitter = CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, data):
        """
        Duyệt qua directory chứa các file txts.
        Sử dụng TextLoader từ langchain và split text trong từng file txt.
        Thêm chỉ mục metadata:
            - source: file_path
            - chunk_index: index của từng chunk trong 1 file txt
            - id: index của file txt trong thư mục data
        """
        docs = []
        for file_id, file_path in enumerate(sorted(data.glob("*.txt"))):
            loader = TextLoader(file_path)
            doc = loader.load()
            split_texts = self.text_splitter.split_documents(doc)
            
            for chunk_id, text in enumerate(split_texts):
                if self._prefix:
                    prefix = file_path.stem
                    prefix = ' '.join(prefix.split("_"))
                    text.page_content = f"{prefix} {text.page_content}"
                    text.metadata["prefix"] = prefix.lower()
                text.metadata["source"] = file_path.name
            docs.extend(split_texts)
            
        return docs
    

class DatabaseMilvus:
    def __init__(self, collection_name, documents, model_name="intfloat/e5-large-v2", host="192.168.20.156", port="19530", hybrid = False, recreate=False):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.URI = f"http://{self.host}:{self.port}"

        print(f"Loading embeddings: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': "cuda"},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Recreate collection if needed
        connections.connect(alias="default", host=self.host, port=self.port)

        if recreate and utility.has_collection(self.collection_name):
            print(f"Dropping existing collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)
        
        print(f"Creating new collection: {self.collection_name}")
        
        self.vector_store = Milvus(
        embedding_function=self.embeddings,
        collection_name=self.collection_name,  
        connection_args={"uri": self.URI},
        auto_id=True,
        index_params={"index_type": "HNSW", "metric_type": "COSINE"},
        )
        
        if hybrid:
            print("Enabling BM25 for Hybrid Search")
            self.vector_store.builtin_function = BM25BuiltInFunction()
            
        batch_size = 4
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]  
            print(f"Upserting batch {i // batch_size + 1} of {len(documents) // batch_size + 1}...")

            self.vector_store.add_documents(batch)

def main():
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    
    data = Path(
        "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/TRNS-AI/data/English_sanitized_policies"
        # "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/TRNS-AI/data/V_X_university"
        # "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/TRNS-AI/data/tvts_data"
    )   
    
    collection_name = data.stem
    chunk_size = 1000
    chunk_overlap = 200
    
    separator = "\n"
    prefix = True
    
    hybrid = True
    if hybrid: collection_name += '_hybrid'
    
    text_splitter = TextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, _prefix=prefix)
    docs = text_splitter.split(data)
    db = DatabaseMilvus(collection_name=collection_name, documents=docs, model_name=model_name, hybrid=hybrid, recreate=True)
    
if __name__ == "__main__":
    collection_name = "English_sanitized_policies_hybrid"
    host="192.168.20.156"
    port="19530"
    connections.connect(alias="default", host=host, port=port)

    if utility.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)
    # main()


