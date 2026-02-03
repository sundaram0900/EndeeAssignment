# Endee RAG Project

A **Retrieval Augmented Generation (RAG)** and **Semantic Search** application powered by [Endee](https://endee.io), a high-performance open-source vector database.

---

## 📋 Project Overview

### Problem Statement

Traditional keyword-based search systems struggle with understanding the *meaning* behind user queries. When users search for information, they often use different words than those present in the source documents, leading to poor search results.

**Example:** A user searching for "how to fix a bug in my code" might miss relevant documents titled "debugging techniques" or "troubleshooting software errors" because the exact keywords don't match.

### Solution

This project implements a **semantic search** system that understands the *meaning* of text rather than just matching keywords. By converting text into numerical vectors (embeddings) that capture semantic meaning, the system can find relevant content even when the exact words differ.

**Key Capabilities:**
- **Ingest** any text content (articles, documentation, notes)
- **Search** using natural language queries
- **Retrieve** semantically similar content with high accuracy

---

## 🏗️ System Design / Technical Approach

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Streamlit Web App)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG ENGINE                                 │
│  ┌─────────────────┐    ┌─────────────────────────────────┐    │
│  │  Text Chunking  │ -> │  Embedding Generation           │    │
│  │  (Split text)   │    │  (all-MiniLM-L6-v2 model)       │    │
│  └─────────────────┘    └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENDEE CLIENT                                │
│         (Python SDK wrapper for API interactions)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ENDEE VECTOR DATABASE                         │
│              (Docker container on port 8080)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Vector Storage (HNSW Index)                          │   │
│  │  • Similarity Search (Cosine Distance)                  │   │
│  │  • Metadata Storage                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Ingestion Flow
1. **User Input** → User pastes text content in the sidebar
2. **Chunking** → Text is split into smaller chunks (~200 words each)
3. **Embedding** → Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
4. **Storage** → Vectors are upserted into Endee with the original text as metadata

#### Search Flow
1. **Query Input** → User enters a natural language question
2. **Query Embedding** → Question is converted to a 384-dimensional vector
3. **Similarity Search** → Endee finds the k-nearest vectors using cosine similarity
4. **Results** → Original text chunks are returned, ranked by relevance

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Convert text to vectors |
| **Vector Database** | Endee | Store and search vectors |
| **Container** | Docker | Run Endee server |
| **Language** | Python 3.9+ | Application logic |

---

## 🔷 How Endee is Used

### What is Endee?

[Endee](https://endee.io) is a high-performance, open-source vector database designed for AI applications. It provides:

- **Low-latency vector search** with high recall
- **HNSW indexing** for efficient approximate nearest neighbor search
- **RESTful API** and native SDKs (Python, TypeScript, Go, Java)
- **Cosine, L2, and Inner Product** distance metrics

### Integration in This Project

We use the official **Endee Python SDK** (`pip install endee`) to interact with the database:

```python
from endee import Endee, Precision

# Initialize client (connects to localhost:8080 by default)
client = Endee()

# Create an index for storing vectors
client.create_index(
    name="rag_documents",
    dimension=384,           # Matches embedding model output
    space_type="cosine",     # Cosine similarity for semantic search
    precision=Precision.INT8D  # Optimized storage
)

# Get index reference
index = client.get_index(name="rag_documents")

# Upsert vectors with metadata
index.upsert([
    {
        "id": "unique-id",
        "vector": [0.1, 0.2, ...],  # 384-dim embedding
        "meta": {"document": "Original text content"}
    }
])

# Query for similar vectors
results = index.query(
    vector=[0.15, 0.25, ...],  # Query embedding
    top_k=5                     # Return top 5 matches
)
```

### Why Endee?

| Feature | Benefit |
|---------|---------|
| **Open Source** | No vendor lock-in, full transparency |
| **High Performance** | Sub-millisecond query latency |
| **Simple API** | Easy integration with Python applications |
| **Docker Ready** | One-command deployment |
| **Quantization** | INT8D precision reduces memory usage |

---

## 🚀 Setup and Execution Instructions

### Prerequisites

- **Docker Desktop** (required for Windows/Mac to run Endee)
- **Python 3.9+**
- **pip** (Python package manager)

### Step 1: Clone or Navigate to Project

```bash
cd endee_rag_project
```

### Step 2: Start Endee Database

Ensure Docker Desktop is running, then:

```bash
docker-compose up -d
```

This starts the Endee vector database on `http://localhost:8080`.

**Verify it's running:**
```bash
docker ps
# Should show: endee-db container running
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` - Web UI framework
- `sentence-transformers` - Embedding model
- `endee` - Official Endee Python SDK
- `numpy` - Numerical operations
- `requests` - HTTP client

### Step 4: Run the Application

```bash
cd src
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Step 5: Using the Application

1. **Ingest Content:**
   - Expand the sidebar (click `>` if collapsed)
   - Paste your text content (articles, documentation, notes)
   - Click **"Ingest Content"**
   - Wait for confirmation message

2. **Search:**
   - Enter a natural language question in the search box
   - Click **"Search"**
   - View semantically relevant results

---

## 📁 Project Structure

```
endee_rag_project/
├── docker-compose.yml     # Endee container configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── src/
    ├── app.py             # Streamlit web application
    ├── endee_client.py    # Endee SDK wrapper
    ├── rag_engine.py      # Text chunking & embedding
    └── verify_rag.py      # Verification script
```

---

## 🔧 Troubleshooting

### Endee Container Won't Start

```bash
# Check container logs
docker logs endee-db

# Restart the container
docker-compose down
docker-compose up -d
```

### "Index not available" Error

The index may not have been created. Run the verification script:

```bash
cd src
python verify_rag.py
```

### Connection Refused on Port 8080

Ensure Docker Desktop is running and the Endee container is healthy:

```bash
docker ps -a
# Look for endee-db with status "Up"
```

---

## 📚 Additional Resources

- [Endee Documentation](https://docs.endee.io)
- [Endee Python SDK Guide](https://docs.endee.io/python-sdk/quickstart)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 License

This project is provided as-is for demonstration purposes.
