# LangChain Semantic Text Splitter

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Embeddings-Google%20Gemini-orange)](https://ai.google.dev/)
[![RAG](https://img.shields.io/badge/Focus-Semantic%20Chunking-purple)](https://python.langchain.com/docs/concepts/text_splitters/)

A practical implementation of semantic text chunking using LangChain's `SemanticChunker` and Google Generative AI Embeddings. This project explores how documents can be divided based on changes in meaning rather than relying only on fixed character lengths.

---

## 🏗️ Project Overview

Text splitting is an essential step in building **Retrieval-Augmented Generation (RAG)** applications.

Traditional text splitters generally divide documents using character counts, separators, or structural boundaries. However, these approaches may sometimes separate related information or group unrelated topics together.

Semantic chunking takes a different approach. It uses embedding representations to understand relationships between consecutive sentences and identify meaningful changes in context.

This project demonstrates how to use LangChain's `SemanticChunker` with Google Generative AI Embeddings to create semantically meaningful document chunks.

## 🛠️ Key Technical Implementations

### 1. SemanticChunker Integration

The project uses LangChain's `SemanticChunker` from `langchain_experimental.text_splitter`.

Unlike basic character-based splitting, semantic chunking uses embeddings to represent sentence meaning and identify significant changes in context.

```python
from langchain_experimental.text_splitter import SemanticChunker

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5
)
```

### 2. Google Generative AI Embeddings

The implementation uses Google's embedding model:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    task_type="CLUSTERING"
)
```

The embedding model converts text into numerical vector representations that can be compared to identify semantic relationships.

The `CLUSTERING` task type is configured for the embedding operation in this implementation.

### 3. Standard Deviation-Based Breakpoint Detection

SemanticChunker supports different breakpoint detection strategies.

This project uses:

```python
breakpoint_threshold_type="standard_deviation"
breakpoint_threshold_amount=1.5
```

The standard deviation method identifies potential topic boundaries by examining how sentence-level semantic distances differ from the average.

The threshold amount controls how sensitive the chunker is to changes in semantic distance.

A lower threshold can produce more frequent splits, while a higher threshold generally produces fewer splits.

### 4. Document Creation

The `create_documents()` method converts the input text into LangChain `Document` objects.

```python
docs = text_splitter.create_documents([sample])

print(len(docs))
print(docs)
```

Each resulting document contains a text chunk and associated metadata.

The implementation prints:
- Total number of generated chunks.
- The generated document objects and their contents.

---

## 🔄 Project Workflow

```text
Input Text
    |
    v
Sentence-Level Processing
    |
    v
Google Generative AI Embeddings
    |
    v
Semantic Distance Analysis
    |
    v
Standard Deviation Breakpoint Detection
    |
    v
Semantic Chunking
    |
    v
LangChain Document Objects
    |
    v
Display Generated Chunks
```

---

## 💻 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | Text processing framework |
| LangChain Experimental | SemanticChunker implementation |
| Google Generative AI | Text embeddings |
| python-dotenv | Environment variable management |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/semantic-meaning-splitter.git
```

### 2. Navigate to the Project Directory

```bash
cd semantic-meaning-splitter
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -U langchain-experimental langchain-google-genai python-dotenv
```

### 5. Configure Your Google API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Replace the example value with your actual Google API key.

**Important:** Never upload your actual API key to GitHub. Add `.env` to your `.gitignore` file.

Example `.gitignore` entry:

```gitignore
.env
venv/
__pycache__/
```

### 6. Run the Implementation

```bash
python semantic_meaning_based.py
```

The script will:
1. Load environment variables.
2. Initialize Google Generative AI Embeddings.
3. Configure the semantic text splitter.
4. Process the sample text.
5. Generate semantic document chunks.
6. Print the number of chunks and their contents.

**Note:** The embedding model must be available to your Google API account. Model availability, API access, and usage limits may vary.

---

## 📚 Why Semantic Chunking Matters in RAG

In a typical RAG pipeline, documents are divided into chunks before generating embeddings and storing them in a vector database.

Poor chunk boundaries can result in:
- Loss of important contextual information.
- Unrelated topics appearing in the same chunk.
- Less relevant retrieval results.
- Incomplete context being passed to an LLM.

Semantic chunking attempts to improve this process by identifying topic changes and grouping related sentences together.

It can be useful for:
- Document-based question answering.
- Knowledge-base search.
- Research paper processing.
- Context-aware document retrieval.
- Building more advanced RAG applications.

---

## 🎯 Learning Outcomes

By exploring this project, you will understand:

- Why semantic chunking is useful in Generative AI.
- How LangChain's `SemanticChunker` works.
- How embedding models represent text.
- The role of semantic distances in identifying topic changes.
- How standard deviation-based breakpoint detection works.
- How to generate LangChain `Document` objects.
- How semantic preprocessing fits into a RAG pipeline.

---

## 🚀 Future Improvements

Possible extensions to this project include:

- Comparing percentile and gradient-based breakpoint methods.
- Experimenting with different standard deviation thresholds.
- Comparing semantic chunking with recursive character splitting.
- Testing different embedding models.
- Processing external documents such as PDFs and Markdown files.
- Integrating generated chunks with a vector database.
- Building a complete RAG question-answering application.

---

## 🔗 Repository

GitHub: https://github.com/mani787060/semantic-meaning-splitter

**Project Focus:** Semantic Text Chunking | LangChain | Google Generative AI Embeddings | RAG Preprocessing
