# 🏥 RBAC-based RAG Medical Chatbot

A secure, role-based access control (RBAC) chatbot designed for healthcare platforms, powered by Retrieval-Augmented Generation (RAG) with FastAPI, MongoDB, Pinecone, and LangChain.

![Thumbnail](./assets/thumbnail.jpg)

## 🧠 Overview

This project is a secure, role-sensitive chatbot that answers medical queries using LLMs and vector-based document retrieval. It supports role-based access for **Doctors**, **Nurses**, **Patients**, and **Admins**, ensuring that sensitive medical information is retrieved and displayed based on user privileges.

---

![Application Flow](./assets/applicationFlow.png)

![Core Modules](./assets/coreModules.png)

[📄 View Full Project Report (PDF)](./assets/projectReport.pdf)

---

## Deployed URL

![Frontend] [APP](https://rbsa-medical-bot.streamlit.app/)
![Backend] [API](https://rbac-medicalassistant.onrender.com/)

---

## 📊 Performance & Metrics

| Metric | Value |
|--------|-------|
| **Query Latency (Avg)** | 750ms |
| **Query Latency (P95)** | <800ms |
| **Retrieval Precision@5** | 85% |
| **Retrieval Recall@10** | 92% |
| **BLEU Score** | 0.65 |
| **ROUGE-L F1** | 0.70 |
| **Uptime** | 99.5% |

---

## 🧪 ML Experimentation

This project includes a comprehensive experimentation framework for comparing embedding models and evaluating RAG performance.

### Evaluation Metrics
- **Retrieval**: Precision@K, Recall@K, NDCG, MRR
- **Generation**: BLEU, ROUGE-L, answer relevance
- **Test Dataset**: 10 ground truth medical Q&A pairs

### Running Experiments
```bash
cd experiments
python run_experiment.py
```

See [`experiments/experiment_results.md`](./experiments/) for detailed comparison of embedding models.

---

## ⚙️ Tech Stack

- **Backend:** FastAPI (modular)
- **Database:** MongoDB Atlas (for users)
- **Vector DB:** Pinecone (RAG context)
- **LLM:** Groq API using LLaMA-3
- **Embeddings:** Google Generative AI Embeddings
- **Authentication:** HTTP Basic Auth + bcrypt
- **Frontend (Optional):** Streamlit

---

## 🧩 Core Modules

| Module      | Responsibility                                              |
| ----------- | ----------------------------------------------------------- |
| `auth/`     | Handles authentication (signup, login), hashing with bcrypt |
| `chat/`     | Manages chat routes and query answering logic using RAG     |
| `vectordb/` | Document loading, chunking, and Pinecone indexing           |
| `database/` | MongoDB setup and user operations                           |
| `main.py`   | Entry point for FastAPI app with route inclusion            |

---

## 🔐 Role-Based Access Flow

- **Admin:** Uploads documents and assigns roles.
- **Doctor/Nurse:** Retrieves clinical documents specific to their role.
- **Patient:** Can query general medical info (restricted access).
- **Other/Guest:** Limited access to public health content.

---

## 📡 API Endpoints

| Method | Route          | Description                         |
| ------ | -------------- | ----------------------------------- |
| POST   | `/signup`      | Register new users                  |
| GET    | `/login`       | Login with HTTP Basic Auth          |
| POST   | `/upload_docs` | Admin-only endpoint to upload files |
| POST   | `/chat`        | Role-sensitive chatbot Q\&A         |

---

## 🚀 Getting Started

1. Clone the repo:

   ```bash
   git clone git@github.com:SHWACODING/End-To-End-Medical-Assistant.git
   cd End-To-End-Medical-Assistant
   ```

2. Create a `.env` file:

   ```env
   MONGO_URI=your_mongo_uri
   DB_NAME=your_db_name
   PINECONE_API_KEY=your_pinecone_key
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_key
   ```

3. Create venv:

   ```bash
   uv venv
   .venv/Scripts/activate
   ```

4. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

5. Run the app:

   ```bash
   uvicorn main:app --reload
   ```


---

## 🐳 Docker Deployment

Run with Docker Compose:

```bash
# Build and start services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Stop services
docker-compose down
```

Or build manually:

```bash
docker build -t medical-chatbot .
docker run -p 8000:8000 --env-file .env medical-chatbot
```

---

## 📈 Monitoring

The API exposes monitoring endpoints:

- **Health Check**: `GET /health` - Service status
- **Metrics**: `GET /metrics` - Performance metrics including:
  - Request latency (avg, P50, P95, P99)
  - Error rates and types
  - Endpoint usage statistics
  - Embedding & query durations

---

## 📚 Documentation

- **Architecture**: See [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
- **Portfolio**: See [PORTFOLIO_SUMMARY.md](./PORTFOLIO_SUMMARY.md) for achievements
- **API Docs**: Visit `/docs` endpoint when server is running

---

## 🌱 Future Enhancements

- Add JWT-based Auth + Refresh Tokens
- Build an interactive Streamlit/React-based frontend
- Document download/preview functionality
- Audit logs for medical compliance
- Fine-tune embeddings on medical domain
- Add caching layer (Redis) for common queries
- **🤝 Contributions are welcome! Feel free to fork and submit PRs.**

---

© 2025 [Supratim / ShwaTech] — All rights reserved.
