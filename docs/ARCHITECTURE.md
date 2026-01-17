# System Architecture - RBAC Medical Chatbot

## Overview

This document describes the technical architecture of the RBAC-based RAG Medical Chatbot, including system design, data flows, ML pipeline, and deployment architecture.

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        A[Streamlit Frontend]
        B[HTTP Clients]
    end
    
    subgraph "API Gateway"
        C[FastAPI Backend]
        D[Auth Middleware]
        E[Monitoring Middleware]
    end
    
    subgraph "Core Services"
        F[Authentication Service]
        G[Document Management]
        H[Chat Service]
    end
    
    subgraph "ML/AI Layer"
        I[Embedding Generator<br/>Sentence Transformers]
        J[Vector Store<br/>Pinecone]
        K[LLM<br/>LLaMA-3 via Groq]
    end
    
    subgraph "Data Layer"
        L[(MongoDB<br/>User Data)]
        M[Document Storage<br/>File System]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
    F --> L
    G --> M
    G --> I
    I --> J
    H --> I
    H --> J
    H --> K
```

---

## Data Flow Diagrams

### 1. Document Upload & Vectorization Flow

```mermaid
sequenceDiagram
    participant Admin
    participant API as FastAPI
    participant FS as File System
    participant Embed as Embedding Model
    participant PDB as Pinecone DB
    
    Admin->>API: POST /upload_docs (PDF files)
    API->>API: Verify admin role
    API->>FS: Save PDF files
    API->>API: Load documents (PyPDFLoader)
    API->>API: Split into chunks (1000 tokens)
    loop For each chunk
        API->>Embed: Generate embedding (768-dim)
        Embed-->>API: Return vector
        API->>API: Add metadata (role, source, page)
    end
    API->>PDB: Upsert vectors with metadata
    PDB-->>API: Confirm indexing
    API-->>Admin: Success response
```

### 2. Query & RAG Response Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Embed as Embedding Model
    participant PDB as Pinecone DB
    participant LLM as LLaMA-3 (Groq)
    
    User->>API: POST /chat (query + auth)
    API->>API: Validate user role
    API->>Embed: Embed query (768-dim)
    Embed-->>API: Query vector
    API->>PDB: Vector search (top_k=10)
    PDB-->>API: Retrieved chunks (with scores)
    API->>API: Filter by user role
    API->>API: Assemble context from chunks
    API->>LLM: Generate answer (query + context)
    LLM-->>API: Generated answer
    API->>API: Extract source citations
    API-->>User: {answer, sources}
```

### 3. Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant MongoDB
    participant bcrypt
    
    rect rgb(200, 220, 250)
        Note over Client,bcrypt: Signup Flow
        Client->>API: POST /signup (username, password, role)
        API->>bcrypt: Hash password
        bcrypt-->>API: Hashed password
        API->>MongoDB: Store user document
        MongoDB-->>API: User created
        API-->>Client: {user_id, username, role}
    end
    
    rect rgb(250, 220, 200)
        Note over Client,bcrypt: Login Flow
        Client->>API: GET /login (HTTP Basic Auth)
        API->>MongoDB: Fetch user by username
        MongoDB-->>API: User document
        API->>bcrypt: Verify password hash
        bcrypt-->>API: Verification result
        alt Valid credentials
            API-->>Client: {user_id, username, role}
        else Invalid credentials
            API-->>Client: 401 Unauthorized
        end
    end
```

---

## ML Pipeline Architecture

### RAG Pipeline Components

```mermaid
graph LR
    A[User Query] --> B[Query Embedding<br/>MiniLM-L6-v2]
    B --> C[Vector Search<br/>Cosine Similarity]
    C --> D[Top-K Retrieval<br/>K=10]
    D --> E[RBAC Filtering<br/>Role-based Access]
    E --> F[Context Assembly<br/>Chunk Joining]
    F --> G[Prompt Engineering<br/>Template Application]
    G --> H[LLM Generation<br/>LLaMA-3.1-8B]
    H --> I[Response Formatting<br/>JSON Output]
    I --> J[User Response]
```

### Embedding Pipeline

```mermaid
graph TB
    A[PDF Documents] --> B[Document Loader<br/>PyPDFLoader]
    B --> C[Text Extraction]
    C --> D[Text Splitter<br/>1000 chars, 100 overlap]
    D --> E[Chunks with Metadata]
    E --> F[Batch Embedding<br/>Sentence Transformers]
    F --> G[768-dim Vectors]
    G --> H[Metadata Attachment<br/>role, source, page, doc_id]
    H --> I[Pinecone Upsert]
    I --> J[Vector Index]
```

---

## Technology Stack Deep Dive

### Backend (FastAPI)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI 0.104+ | Async API routes, auto docs |
| **ASGI Server** | Uvicorn | Production-grade async server |
| **Routing** | APIRouter | Modular route organization |
| **Validation** | Pydantic | Request/response schemas |
| **Auth** | HTTP Basic Auth | Simple, secure authentication |
| **Password Hashing** | bcrypt | Secure password storage |

### AI/ML Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangChain | RAG pipeline framework |
| **Embeddings** | sentence-transformers | Text → vector conversion |
| **Vector DB** | Pinecone | Semantic search engine |
| **LLM** | Groq API (LLaMA-3.1) | Answer generation |
| **Document Parsing** | PyPDF | PDF text extraction |

### Data Storage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **User Database** | MongoDB Atlas | User accounts & auth |
| **Vector Storage** | Pinecone | Semantic embeddings |
| **File Storage** | Local File System | Original PDF files |

### Deployment

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Hosting** | Render | Cloud platform (PaaS) |
| **Frontend Hosting** | Streamlit Cloud | Frontend deployment |
| **Containerization** | Docker | Reproducible deployments |
| **Orchestration** | Docker Compose | Multi-service management |

---

## Database Schemas

### MongoDB User Collection

```python
{
  "_id": ObjectId,
  "username": str,           # Unique username
  "hashed_password": str,    # bcrypt hashed password
  "role": str,               # "admin" | "doctor" | "nurse" | "patient" | "other"
  "created_at": datetime,
  "last_login": datetime
}
```

### Pinecone Vector Metadata

```python
{
  "id": str,                 # "{doc_id}-{chunk_index}"
  "vector": [float...],      # 768-dim embedding
  "metadata": {
    "text": str,             # Original chunk text
    "source": str,           # Filename (e.g., "diabetes_guide.pdf")
    "doc_id": str,           # Unique document ID
    "role": str,             # Access level ("doctor", "nurse", "patient", "public")
    "page": int              # Page number in original document
  }
}
```

---

## API Endpoints

### Authentication

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/signup` | POST | No | Register new user |
| `/login` | GET | Yes (Basic) | Authenticate user |

**Request Example**:
```json
POST /signup
{
  "username": "dr_smith",
  "password": "secure_password",
  "role": "doctor"
}
```

### Document Management

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/upload_docs` | POST | Yes (Admin only) | Upload & vectorize documents |

**Request Example**:
```bash
curl -X POST http://localhost:8000/upload_docs \
  -H "Authorization: Basic YWRtaW46cGFzc3dvcmQ=" \
  -F "files=@document.pdf" \
  -F "role=doctor" \
  -F "doc_id=doc_12345"
```

### Chat

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/chat` | POST | Yes | Query the RAG chatbot |

**Request Example**:
```json
POST /chat
{
  "query": "What are the symptoms of diabetes?"
}
```

**Response Example**:
```json
{
  "answer": "Common symptoms of diabetes include...",
  "sources": ["diabetes_guide.pdf", "clinical_handbook.pdf"]
}
```

### Monitoring

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/health` | GET | No | Health check |
| `/metrics` | GET | No | Performance metrics |

---

## Security Architecture

### Authentication & Authorization

```mermaid
graph TB
    A[Incoming Request] --> B{Has Auth Header?}
    B -->|No| C[401 Unauthorized]
    B -->|Yes| D[Extract Credentials]
    D --> E[Query MongoDB]
    E --> F{User Exists?}
    F -->|No| C
    F -->|Yes| G[Verify bcrypt Hash]
    G --> H{Password Match?}
    H -->|No| C
    H -->|Yes| I[Extract User Role]
    I --> J{Endpoint Requires Admin?}
    J -->|Yes| K{User Role = Admin?}
    K -->|No| L[403 Forbidden]
    K -->|Yes| M[Proceed to Handler]
    J -->|No| M
    M --> N[Apply RBAC Filtering]
    N --> O[Return Response]
```

### RBAC Document Filtering

| User Role | Access Level | Description |
|-----------|--------------|-------------|
| **Admin** | All documents | Can upload and access all content |
| **Doctor** | Doctor + Public | Clinical documents and general health |
| **Nurse** | Nurse + Public | Nursing protocols and general health |
| **Patient** | Public only | General health information |
| **Other/Guest** | Public only | Limited to public health content |

**Implementation**:
```python
# In vector search
for match in results["matches"]:
    doc_role = match["metadata"]["role"]
    if user_role == "admin" or doc_role == user_role or doc_role == "public":
        # Include this document
        filtered_contexts.append(match["metadata"]["text"])
```

---

## Performance Optimizations

### 1. Async Operations
- All I/O operations (DB queries, API calls) use asyncio
- Pinecone SDK wrapped with `asyncio.to_thread()` for non-blocking execution
- Concurrent embedding generation for batches

### 2. Connection Pooling
- Lazy initialization of Pinecone client
- MongoDB connection pooling via PyMongo
- Reusable HTTP sessions for API calls

### 3. Caching Strategy
```python
# Global instance caching
_pinecone_index = None

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index("medical-rag-index")
    return _pinecone_index
```

### 4. Batch Processing
- Documents split into chunks before embedding
- Batch upsert to Pinecone (reduced API calls)
- Progress bars for long-running operations (tqdm)

---

## Monitoring & Observability

### Metrics Tracked

```python
{
  "total_requests": 1543,
  "total_errors": 12,
  "error_rate": 0.0078,
  "avg_request_duration_ms": 750,
  "p50_request_duration_ms": 680,
  "p95_request_duration_ms": 1200,
  "p99_request_duration_ms": 1850,
  "avg_embedding_duration_ms": 120,
  "avg_query_duration_ms": 45,
  "endpoint_counts": {
    "/chat": 1200,
    "/upload_docs": 50,
    "/login": 293
  },
  "error_types": {
    "ValidationError": 8,
    "PineconeException": 4
  }
}
```

### Logging Strategy

| Level | Use Case | Example |
|-------|----------|---------|
| **INFO** | Request/response tracking | "→ POST /chat - 200 - 0.75s" |
| **WARNING** | Degraded performance | "Embedding took 5.2s (expected <2s)" |
| **ERROR** | Operation failures | "Pinecone query failed: timeout" |
| **DEBUG** | Development details | "Retrieved 10 chunks, filtered to 7" |

---

## Deployment Architecture

### Production Setup (Render + Streamlit Cloud)

```mermaid
graph TB
    subgraph "User Traffic"
        A[End Users]
    end
    
    subgraph "Streamlit Cloud"
        B[Streamlit Frontend]
    end
    
    subgraph "Render"
        C[FastAPI Backend]
        D[Docker Container]
    end
    
    subgraph "External Services"
        E[(MongoDB Atlas)]
        F[(Pinecone)]
        G[Groq API]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    D --> G
```

### Local Development Setup (Docker Compose)

```mermaid
graph TB
    subgraph "Local Machine"
        A[Docker Host]
        
        subgraph "Docker Network"
            B[Backend Container<br/>Port 8000]
            C[Optional: MongoDB<br/>Port 27017]
        end
        
        D[Volumes<br/>uploaded_docs/]
    end
    
    E[Developer]
    F[(External: Pinecone)]
    G[External: Groq API]
    
    E --> A
    B --> C
    B --> D
    B --> F
    B --> G
```

### Environment Variables

```bash
# MongoDB
MONGO_URI=mongodb+srv://...
DB_NAME=medical_chatbot

# Pinecone
PINECONE_API_KEY=pcsk_xxxxx

# AI Services
GOOGLE_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_xxxxx
```

---

## Scalability Considerations

### Current Capacity
- **Concurrent Users**: ~100 (Render free tier)
- **Vector Storage**: 10,000+ embeddings (Pinecone free tier: 100K vectors)
- **Request Throughput**: ~50 req/min

### Scaling Strategies

1. **Horizontal Scaling**
   - Deploy multiple backend instances behind load balancer
   - Stateless design enables easy replication

2. **Database Scaling**
   - MongoDB Atlas auto-scaling
   - Pinecone supports millions of vectors with same API

3. **Caching Layer**
   - Add Redis for frequently asked questions
   - Cache embeddings for common queries

4. **CDN Integration**
   - Serve static assets via CDN
   - Reduce frontend load times

---

## Future Enhancements

1. **Advanced Authentication**
   - JWT-based auth with refresh tokens
   - OAuth integration (Google, Microsoft)

2. **Enhanced Monitoring**
   - Prometheus + Grafana dashboards
   - Distributed tracing (Jaeger)

3. **ML Improvements**
   - Fine-tuned domain-specific embeddings
   - Multi-model ensemble for critical queries
   - Active learning from user feedback

4. **Compliance & Audit**
   - HIPAA compliance features
   - Audit logs for all access
   - Data encryption at rest

---

## References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- LangChain Docs: https://python.langchain.com/
- Pinecone Guides: https://docs.pinecone.io/
- Sentence Transformers: https://www.sbert.net/

---

*Last Updated: January 2026*
