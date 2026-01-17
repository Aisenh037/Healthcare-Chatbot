# Medical RAG Assistant (Full-Stack MLOps Project)

> **Role**: AI Solutions Developer / Full-Stack ML Engineer  
> **Tech Stack**: Python, FastAPI, Streamlit, Docker, Groq/Llama-3 (LLM), RBAC Security

## 🎯 Project Overview
An **Industry-Grade Retrieval-Augmented Generation (RAG)** platform designed for healthcare professionals. It features a secure, role-based architecture where Admins, Doctors, and Patients interact with a centralized medical knowledge base.

Unlike basic notebooks, this project demonstrates a **Production-Ready Microservices Architecture**:
- **Core Engine**: A standalone Python RAG library (`rag_engine`).
- **API Gateway**: A modular FastAPI backend with Auth & RBAC (`server`).
- **Client UI**: A professional Streamlit dashboard (`client`).

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "Frontend Layer"
        UI[Streamlit Client] -->|Login/Chat| API[FastAPI Gateway]
    end

    subgraph "Backend Service (Port 8000)"
        API -->|Auth| Auth[Auth Module (RBAC)]
        API -->|Upload| Docs[Doc Processing]
        API -->|Query| Chat[Chat Controller]
    end
    
    subgraph "Core Logic (rag_engine)"
        Docs -->|Index| Engine[RAG Engine]
        Chat -->|Retrieve| Engine
        Engine -->|Context| LLM[Groq / Llama-3]
    end
```

## 🚀 Key Features

1.  **Role-Based Access Control (RBAC)**:
    *   **Admins**: Full access to upload/manage documents.
    *   **Doctors**: Access to detailed clinical answers.
    *   **Patients**: Access to simplified health info.
2.  **Modular Design**: The AI logic (`rag_engine`) is completely decoupled from the API (`server`) and UI (`client`).
3.  **Persistant Zero-DB**: Uses a clever file-based mock database for Users and Vectors, ensuring data survives restarts without complex database setup (perfect for portfolios).

## 🛠️ How to Run

### **Prerequisites**
- Python 3.9+
- API Key (Groq or xAI)

### **1. Start the Backend API**
```bash
# Sets up the Brain 🧠
uvicorn server.main:app --host 0.0.0.0 --port 8000
```
*   **Documentation**: http://localhost:8000/docs
*   **Health Check**: http://localhost:8000/health

### **2. Start the Client UI**
```bash
# Sets up the Face 🖥️
streamlit run client/main.py
```
*   **URL**: http://localhost:8501 (or 8502)

### **🔐 Default Credentials**
| Role | Username | Password | Access |
|------|----------|----------|--------|
| **Admin** | `admin` | `admin123` | Uploads & Chat |
| **Doctor** | `doctor` | `doc123` | Chat (Clinical) |
| **Patient** | `patient` | `patient123` | Chat (Basic) |

## 📁 Code Structure (Industry Grade)

```
├── rag_engine/            # The Core AI Logic (Library)
│   ├── 1_load...          # ETL Pipeline
│   ├── 3_store...         # Vector Database
│   └── engine.py          # Unified Interface
│
├── server/                # The API Gateway (FastAPI)
│   ├── auth/              # JWT/Basic Auth & RBAC
│   ├── chat/              # Query Processing
│   ├── docs/              # Document Ingestion
│   └── main.py            # App Entry Point
│
├── client/                # The Frontend (Streamlit)
│   └── main.py            # Dashboard & UI
│
└── data/                  # Persistence Layer (JSON DBs)
```

## 💡 Engineering Decisions

*   **Why Microservices?**
    *   "Separating the UI (`client`) from the API (`server`) allows me to scale them independently. I could easily swap Streamlit for a React Native mobile app without touching the backend."
*   **Why Custom RBAC?**
    *   "Security is critical in specific domains. I implemented a middleware-based permission system to demonstrate how to handle sensitive data access."

---
*Created for AI Internship Portfolio*
