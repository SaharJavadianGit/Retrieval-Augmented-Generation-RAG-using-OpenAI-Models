# Retrieval-Augmented-Generation-RAG-using-OpenAI-Models
# 🔎 Retrieval Augmented Generation (RAG) using OpenAI Models

# 📌 Project Overview

This project demonstrates how to build a **Retrieval Augmented
Generation (RAG) pipeline from scratch using OpenAI models**.

The goal of the project is to show how an LLM can answer questions using
**external documents instead of relying only on its training data**.

The system:

1.  Loads a document
2.  Splits the document into chunks
3.  Converts text chunks into embeddings
4.  Stores embeddings in a vector database (FAISS)
5.  Retrieves the most relevant chunks for a question
6.  Sends the retrieved context to an OpenAI GPT model
7.  Generates a final answer

This implementation is intentionally **simple and beginner friendly** so
the internal mechanics of RAG are easy to understand.

------------------------------------------------------------------------

# 🧠 What is Retrieval Augmented Generation (RAG)?

Retrieval Augmented Generation is a technique that combines:

**Information Retrieval + Large Language Models**

Instead of asking the model directly, the system first retrieves
relevant information from a document database.

The LLM then generates an answer **based on the retrieved context**.

This makes AI systems:

-   more accurate
-   able to use private data
-   more reliable
-   less likely to hallucinate

------------------------------------------------------------------------

# ⚙️ RAG Pipeline Architecture

Document\
↓\
Chunking\
↓\
Embeddings (OpenAI)\
↓\
Vector Index (FAISS)\
↓\
User Question\
↓\
Similarity Search\
↓\
Relevant Chunks\
↓\
GPT Model Generates Answer

------------------------------------------------------------------------

# 🤖 OpenAI Models Used

This project uses **OpenAI hosted models**, which are accessed via API.

Models used:

Embedding Model\
`text-embedding-3-small`

Language Model\
`gpt-4o-mini`

⚠️ These models are **different from Hugging Face models**.

  OpenAI Models                Hugging Face Models
  ---------------------------- -----------------------------
  Hosted via API               Usually run locally
  No model download required   Requires downloading models
  Managed by OpenAI            User manages infrastructure
  Simple API calls             Requires transformers setup

------------------------------------------------------------------------

# 📂 Project Structure

rag-openai-project/

│ ├── Project2.ipynb ├── History_of_Art.txt ├── README.md │ └──
requirements.txt

------------------------------------------------------------------------

# 🔐 API Key Security

The OpenAI API key is **not included in this repository** for security
reasons.

Set your key as an environment variable:

``` bash
export OPENAI_API_KEY="your_api_key_here"
```

The code reads the key using:

``` python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Never commit API keys to GitHub.

------------------------------------------------------------------------

# ▶️ Running the Project

Install dependencies:

``` bash
pip install openai faiss-cpu numpy
```

Open the notebook:

``` bash
jupyter notebook Project2.ipynb
```

Run all cells to execute the full RAG pipeline.

------------------------------------------------------------------------

# 🧾 Code Explanation

## 1️⃣ Import Libraries

The project imports required libraries:

-   **OpenAI** → access GPT models and embeddings
-   **FAISS** → vector similarity search
-   **NumPy** → numerical operations
-   **OS** → access environment variables

------------------------------------------------------------------------

## 2️⃣ Load the Document

The system loads a text file:

``` python
with open("History_of_Art.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

This document becomes the knowledge source for answering questions.

------------------------------------------------------------------------

## 3️⃣ Chunk the Document

Large text is split into smaller chunks.

Why?

-   Improves embedding quality
-   Improves retrieval accuracy
-   Fits model context limits

Chunks are stored in a list.

------------------------------------------------------------------------

## 4️⃣ Generate Embeddings

Each chunk is converted into a vector using OpenAI embeddings.

``` python
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)
```

These vectors represent the **semantic meaning of the text**.

------------------------------------------------------------------------

## 5️⃣ Store Vectors in FAISS

Vectors are stored in a FAISS index.

``` python
index = faiss.IndexFlatL2(len(chunk_vectors[0]))
index.add(matrix)
```

FAISS enables **fast similarity search across vectors**.

------------------------------------------------------------------------

## 6️⃣ User Question

A question is provided by the user:

``` python
query = "What were the main ideas in the history of art document?"
```

The query is converted into an embedding.

------------------------------------------------------------------------

## 7️⃣ Retrieve Relevant Chunks

FAISS finds the most similar chunks.

``` python
D, I = index.search(query_vector, k)
```

These retrieved chunks form the **context for the LLM**.

------------------------------------------------------------------------

## 8️⃣ Build the Prompt

The retrieved chunks are combined into a context block.

``` python
context = "\n\n".join(retrieved_chunks)
```

The model is instructed to answer **only using the provided context**.

------------------------------------------------------------------------

## 9️⃣ Generate Final Answer

The prompt is sent to the OpenAI GPT model.

``` python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...]
)
```

The model generates the final response based on the retrieved knowledge.

------------------------------------------------------------------------

# 📊 Key Takeaways

This project demonstrates:

-   How RAG systems work internally
-   How embeddings represent meaning
-   How vector search retrieves relevant information
-   How LLMs generate answers using retrieved context

Using OpenAI models simplifies integration while still providing
powerful language understanding.

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible improvements include:

-   Building a web interface
-   Adding multi-document retrieval
-   Implementing streaming responses
-   Adding a vector database like Pinecone or Weaviate
-   Implementing conversational RAG

------------------------------------------------------------------------

# 👨‍💻 Author

Retrieval Augmented Generation Project\
OpenAI LLM + FAISS Vector Search

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving it a star on
GitHub!
