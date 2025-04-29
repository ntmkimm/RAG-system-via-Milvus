
# Milvus + LangChain System for Document Search and Answering

This system integrates Milvus for document search and a custom Qwen model for generating answers. The documents are preprocessed and embedded using a hybrid BM25 and cosine similarity approach. The setup uses LangChain for document splitting, retrieval, and question answering (QA).

## System Components

1. **Text Preprocessing and Embedding:**
   - **Input:** A folder containing one or more `.txt` files.
   - **Text Splitter:** Splits each document into smaller chunks for better retrieval performance.
   - **Embedding Model:** Uses the `Alibaba-NLP/gte-Qwen2-1.5B-instruct` model for embedding each chunk into vector space.
   - **Vector Store:** Documents are stored in Milvus for efficient vector-based retrieval.
   - Implements **BM25** hybrid search with **cosine similarity** for document retrieval from Milvus.
2. **Question Answering System:**
   - The system utilizes the `Qwen/Qwen2.5-1.5B-Instruct` model to generate answers based on retrieved documents.

## Installation

To set up the environment and run the system, follow the steps below.

### Requirements

```bash
conda create -n rag python=3.10
conda activate rag
pip install torch
pip install langchain
pip install pymilvus
pip install sentence-transformers
pip install transformers
pip install langchain_milvus
```
### Pretrained models
Models is auto download via Hugging face hub or you can manually download the models (`Alibaba-NLP/gte-Qwen2-1.5B-instruct` and `Qwen/Qwen2.5-1.5B-Instruct`)

### Setting Up Milvus and Upload database

Install Milvus and start the server on your machine. You can follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md).
Ensure the Milvus server is running at the specified host and port.
Set 'hybrid=True' if you want to use BM25 for searching.
```bash
CUDA_VISIBLE_DEVICES=0,1 python core.py
```
3. The system will load the documents, split them, embed them, and store them in the Milvus database. After that, it will be ready for querying the system.

### Running the System
Start the system:

```bash
CUDA_VISIBLE_DEVICES=0,1 python system.py
```

Once the system is running, you can input queries to retrieve answers based on the indexed documents.

```bash
Input query: "Can I park at X university?"
```

The system will retrieve relevant documents and use the `Qwen` model to generate a precise answer.

## Configuration

- **Document Folder Path:** In the code, set the path to the folder containing the `.txt` files.
- **Model Settings:** Modify the model names (`embed_model`, `llm_model`) based on your needs.
- **Hybrid Search:** Set `hybrid = True` in the `DatabaseMilvus` constructor to enable BM25 hybrid search.


## Acknowledgments

- **Milvus**: For efficient vector storage and retrieval.
- **LangChain**: For the chaining of language models, document retrieval, and prompt handling.
- **HuggingFace Transformers**: For pre-trained models used for embeddings and question answering.
