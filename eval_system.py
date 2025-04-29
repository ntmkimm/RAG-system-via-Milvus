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

from system import *

class EvalSystem(System):
    def __init__(self, 
                 prompt='',
                 collection_name='',
                 embed_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct", 
                 test_model="Qwen/Qwen2.5-1.5B-Instruct",
                 llm_model="Qwen/Qwen2.5-3B-Instruct",
                 host="192.168.20.156",
                 port="19530"
                 ):
        super().__init__(prompt=prompt, embed_model=embed_model, llm_model=llm_model, collection_name=collection_name, host=host, port=port)
        if test_model != llm_model:
            self.test_llm = load_llm(model_file=test_model)[0]
        else:
            self.test_llm = self.llm    
        self.test_template = """You are an AI assistant.

You have the following context:

{context}

Here is the question:

{question}

And the provided answer based on the context:

{answer}

Your task:
- The answer may be incorrect. Verify whether the provided answer is accurate and fully addresses the question based solely on the given context.
- If the answer is correct, respond with "Yes, the answer is correct."
- If the answer is incorrect, unclear, or incomplete, respond with "No, the answer is incorrect," and provide a correct of answer.
- Do not use any external knowledge beyond the given context.
- Keep your response concise and to the point.
"""
        self.test_prompt = ChatPromptTemplate.from_template(self.test_template)
        
        self.tester = (
            self.test_prompt | self.test_llm  | StrOutputParser()
        )
        
    def parse_llm_output(self, text):

        pattern = r"Context:\s*(.*?)\s*Question:.*?Answer:\s*(.*)"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return {
                "context": match.group(1).strip(),
                "answer": match.group(2).strip()
            }
        return None

    def eval(self, query):
        outputs = self.generate_origin(query)
        outputs = self.parse_llm_output(outputs)
        final_contexts = self.tester.invoke({"context": outputs['context'], "question": query, "answer": outputs['answer']})
        print(final_contexts)
        
        
        
def main():
    hybrid = True
    collection_name = "V_X_university"
    if hybrid: collection_name += '_hybrid'
    
    embed_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    llm_model = "Qwen/Qwen2.5-3B-Instruct"
    test_model = "Qwen/Qwen2.5-3B-Instruct"
    
    
# ========================= NOTE ==================
# prompt: prompt để có output cho câu trả lời đầu tiên
# muốn chỉnh prompt để đánh giá lại kết quả thì vào class EvalSystem
# ĐỪNG CHỈNH 5 DÒNG NÀY KHÔNG THÔI LÀ KHÔNG PARSE ĐƯỢC CÂU TRẢ LỜI

# Context: 
# {context}  

# Question:
# {question}  

# Answer:
    
    prompt = """You are an AI assistant. Your goal is to answer questions accurately and concisely.

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


    system = EvalSystem(prompt=prompt, test_model = test_model, embed_model=embed_model, llm_model=llm_model, collection_name=collection_name)
    
    questions = [
        "is it prohibit to use phone in X university",
        "im from X university, if i go to school by car, where can i park ?",
        "im from university Z, i have my grades is 7. but i miss 30% attendace. how much grade i have?"
        ]
    
    
    for i, q in enumerate(questions):
        print("\n\n\nQuestion ", i + 1)
        system.eval(q)
    
    while True:
        query = input("Input new query: ")
        system.eval(query)
        

if __name__ == "__main__":
    main()
