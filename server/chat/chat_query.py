import asyncio
from typing import Dict, List
# Import our custom engine using the bridge
from rag_engine.engine import MinimalRAGChatbot

# Global singleton instance
chatbot = MinimalRAGChatbot()

async def answer_query(query: str, user_role: str) -> Dict:
    """
    Generate answer using the Minimal RAG engine, respecting user roles.
    
    Args:
        query: User's question
        user_role: Role from JWT/Auth (admin, doctor, nurse, patient)
        
    Returns:
        Dict with answer and sources
    """
    print(f"💬 Processing query for role: {user_role}")
    
    # 1. Check permissions / Role Logic
    # (For MVP: All roles can access, but we log it. 
    # In production, Patients might be restricted to specific categories)
    
    if user_role not in ["admin", "doctor", "nurse", "patient"]:
        return {
            "answer": "Unauthorized role.",
            "sources": []
        }

    # 2. Use the RAG Engine
    # Note: chatbot.ask is synchronous, so we run it in a thread to keep FastAPI async
    try:
        result = await asyncio.to_thread(chatbot.ask, query)
        
        # 3. Role-Based Customization (The "Art")
        # Example: Doctors get technical details, Patients get simplified ones.
        # For this MVP, we'll just tag the response.
        
        final_answer = result['answer']
        
        # Add role-specific context if needed (demo feature)
        if user_role in ["doctor", "nurse"]:
            prefix = "[Medical Professional Access]\n"
        else:
            prefix = ""

        return {
            "answer": prefix + final_answer,
            "sources": result['sources'],
            "role_used": user_role
        }
        
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        return {
            "answer": "I encountered an error processing your request.",
            "sources": []
        }
