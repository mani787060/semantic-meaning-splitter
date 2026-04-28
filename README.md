# LangChain Semantic Text Splitter
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Advanced RAG](https://img.shields.io/badge/Pipeline-Semantic%20Chunking-purple)](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

## 🏗️ Project Overview
This repository tackles the most advanced tier of **Data Chunking** in Retrieval-Augmented Generation (RAG). Traditional splitters cut text based on character counts or markdown syntax. This project implements LangChain's **`SemanticChunker`**, which uses an AI embedding model to analyze sentences, calculate their cosine distance, and group them together mathematically. It only splits a document when it detects a significant shift in the underlying topic, ensuring the LLM receives hyper-focused, cohesive context.

---

## 🛠️ Key Technical Implementations

### 1. Embedding-Driven Segmentation
* **The `SemanticChunker` Integration:** Unlike standard splitters, this tool requires an active LLM embedding connection. It works by first splitting text into sentences, embedding those sentences into vectors, and then comparing the distances between them to map out the semantic flow of the document.

### 2. Dynamic Breakpoint Thresholds
* **Mathematical Splitting Logic:** This implementation demonstrates how to control exactly *when* a split occurs using statistical thresholds:
  * **Percentile:** Splits when the semantic distance between sentences is in the top X percentile of all distances in the document.
  * **Standard Deviation:** Splits when the distance exceeds a certain number of standard deviations from the mean.
  * **Gradient/Interquartile:** Advanced methods for detecting abrupt changes in context.

### 3. High-Fidelity Retrieval Readiness
* **Noise Reduction:** By ensuring every chunk focuses on a single, continuous thought, this pipeline prevents "topic dilution," which occurs when a single vector chunk contains parts of two completely unrelated subjects, confusing the downstream LLM.

---

## 💻 Tech Stack
* **Language:** Python 3.9+
* **Framework:** LangChain Experimental (`langchain-experimental`)
* **Embeddings:** Google GenAI Embeddings (`langchain-google-genai`)
* **Environment:** `python-dotenv`

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/langchain-semantic-meaning-splitter.git](https://github.com/your-username/langchain-semantic-meaning-splitter.git)

2. **Install Dependencies:**
   ```bash
   pip install -U langchain langchain-experimental langchain-google-genai python-dotenv

3. **Setup API Key:**
   Create a .env file in the root directory:

  Plaintext
  GOOGLE_API_KEY=your_gemini_key_here
 
4. **Run the Implementation:**
   ```bash
   python semantic_meaning_based.py   
