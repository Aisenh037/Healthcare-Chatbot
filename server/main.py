from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path

from .auth.routes import router as auth_router, authenticate
from .docs.routes import router as docs_router
from .chat.routes import router as chat_router
from .monitoring.monitoring import RequestLogger, metrics_collector
from .chat.chat_query import chatbot  # Import the global chatbot instance

app = FastAPI(
    title="Medical Assistant RAG API",
    description="RBAC-based medical chatbot with RAG",
    version="1.0.0"
)

# Add monitoring middleware
app.add_middleware(RequestLogger)

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(docs_router, tags=["Documents"]) 
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "medical-assistant-api", "engine": "MinimalRAG"}

@app.get("/metrics")
def get_metrics():
    """Get performance metrics for monitoring"""
    return JSONResponse(content=metrics_collector.get_metrics())

# === KEY UPGRADE: Admin-Only Upload Endpoint ===
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user = Depends(authenticate)  # 🔒 RBAC Protection
):
    """
    Upload and index a medical PDF.
    Only 'admin' role allowed.
    """
    # 1. Check Role
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied. Only Admins can upload documents.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_path = Path(f"data/uploads/{file.filename}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Index the document using the RAG Engine
    # (The chatbot instance handles the heavy lifting)
    try:
        chatbot.load_documents([str(temp_path)])
        return {
            "filename": file.filename, 
            "status": "indexed", 
            "total_chunks": chatbot.vector_store.count(),
            "uploaded_by": user["username"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
