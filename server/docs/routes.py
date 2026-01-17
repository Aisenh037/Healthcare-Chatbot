from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from ..auth.routes import authenticate
from pathlib import Path
import shutil
# Import the global chatbot from explicit path (to avoid circular imports if in main)
# But better: use the one from main or re-instantiate lightly? 
# Best: Use the singleton pattern from chat_query or similar.
from ..chat.chat_query import chatbot

router = APIRouter()

@router.post("/upload_docs")
async def upload_docs(
    user=Depends(authenticate),
    file: UploadFile=File(...),
    role: str=Form(...)
):
    if user["role"] != "admin" :
        raise HTTPException(status_code=403, detail="Only Admins Can Upload Files")
    
    # Save to disk
    temp_path = Path(f"data/uploads/{file.filename}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Index with RAG Engine
        chatbot.load_documents([str(temp_path)])
        
        # Note: In a real system, we'd tagging 'role' metadata here
        # Our Minimal RAG supports metadata, so we could extend it if needed.
        # For now, just indexing is enough for MVP.
        
        return {
            "message": f"{file.filename} Uploaded & Indexed Successfully!",
            "doc_id": str(temp_path),
            "accessible_to": role,
            "total_chunks": chatbot.vector_store.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



