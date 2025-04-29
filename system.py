import os
import re
import torch
from operator import itemgetter
from pymilvus import connections, Collection, MilvusClient
from collections.abc import Sequence 

# LangChain Components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain.load import dumps, loads

# Transformers & Tokenization
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from core import *

def fusion(documents: list[list], k = 60):
            fused_scores = {}
            
            for docs in documents:
                for rank, doc in enumerate(docs):
                    # Convert the document to a string format to use as a key (documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                        
                    fused_scores[doc_str] += 1 / (rank + k) # RRF formula: 1 / (rank + k)

            # Sort in descending order
            reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

            return reranked_results[:15]

def load_llm(model_file):
    print(f"Loading LLM model: {model_file}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_file,
        device_map="auto",          
        torch_dtype=torch.float16,  
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_file,
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.01,
        device_map="auto"  
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm, model, tokenizer

class System:
    def __init__(self, 
                 prompt='',
                 collection_name='',
                 embed_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct", 
                 llm_model="Qwen/Qwen2.5-1.5B-Instruct",
                 host="192.168.20.156",
                 port="19530", 
                 ):
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.URI = f"http://{self.host}:{self.port}"
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True,
                           'convert_to_tensor': True}
        )
        
        self.vector_store = Milvus( 
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.URI},
            index_params={"index_type": "HNSW", "metric_type": "COSINE"},
        )        
        
        # Load retriever and LLM Model and prompt
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 7})
        self.llm, self.model, self.tokenizer = load_llm(llm_model)
        self.model.eval()
        
        # prompt for all task
        self.template = """You are an AI assistant. Your goal is to answer questions accurately and concisely.

Use the provided context and your own knowledge to craft a precise response.

Example
Q: "This semester, I scored 8 points on the final exam for the DSA course. However, I was absent for the lab exam. Can I still get a B in this course?"  
A: "No. Because you missed the lab exam, you received a score of 0 for lab work. According to Regulation #13 of X University, a student with 0 lab points cannot pass the course."

Instructions
- Provide a short and clear answer with an easy-to-read explanation. Using only the regulations relevant to the university that mentioned in question.
- No need further or any extend explaination or information.    

Context: 
{context}  

Question:
{question}  

Answer:

"""
        if prompt != "":
            self.template = prompt
        self.prompt = ChatPromptTemplate.from_template(template=self.template)
        
        # origin rag chain
        self.origin_rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff",  
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt}  
        )
        
        # multi query rag chain + fusion
        self.multi_query_template = """"Your task is just provide 3 sentences that have the same meaning with: '{question}'. Each version should be clear and splitted by a new line. No explaination or furthur information needed"""
        self.prompt_multi_query = ChatPromptTemplate.from_template(self.multi_query_template)

        self.generate_queries = (
            self.prompt_multi_query  | self.llm  |   StrOutputParser()  |  (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()][-3:]) 
            )
        
        self.multi_query_rag_chain = (
            {"context": lambda x: self.get_contexts_by_multi_query(x["question"]), "question": itemgetter("question")} 
            | self.prompt | self.llm  | StrOutputParser()
        )
        
    def get_contexts_by_multi_query(self, query):
        queries = self.generate_queries.invoke({"question": query})
        retrieved_docs = [self.retriever.invoke(q) for q in queries]
        contexts = fusion(retrieved_docs)

        return contexts
    
    
    def generate_by_multi_query(self, query):
        results = self.multi_query_rag_chain.invoke({"question": query})
        return results
    
    def generate_origin(self, query):
        results = self.origin_rag_chain({"query": query})
        return results['result']

    def generate(self, query):
        res = self.generate_by_multi_query(query)
        print(res)
        
def main():
    # cái hybrid này tích hợp vào lúc add embedding rồi, kiểu data sẽ khác (theo đánh giá là retrieval ổn định hơn nếu hybrid (có tích hợp bm25)!!!!!, code ở system không thay đổi, khác ở tên data collection_name thoaiiii
    hybrid = True
    
    collection_name = "V_X_university"
    if hybrid: collection_name += '_hybrid'
    
    embed_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    llm_model = "Qwen/Qwen2.5-3B-Instruct"
    
    system = System(embed_model=embed_model, llm_model=llm_model, collection_name=collection_name)
    
    questions = [
        "is it prohibit to use phone in X university",
        "im from X university, if i go to school by car, where can i park ?",
        "im from university Z, i have my grades is 7. but i miss 30% attendace. how much grade i have?",
        "im from university Z, i had gpa of 3.75 in last semester, but i missed attendance around 40% in math class, now it reduce 3% my gpa. Can i enroll more than 21 credit hours in next semester",
        # "if a hosting campus event, but i cannot control crowd on that event it lead to accident. do i qualify for hosting next campus envent"
        ]
    
    for i, q in enumerate(questions):
        print("\n\n\nQuestion ", i + 1)
        system.generate(q)
    
    while True:
        query = input("Input query: ")
        system.generate(query)
        

if __name__ == "__main__":
    main()
