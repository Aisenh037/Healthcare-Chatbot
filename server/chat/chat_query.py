import asyncio
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

prompt=PromptTemplate.from_template("""
You are a helpful healthcare assistant.Answer the following question
based only on the provided context.

    Question:{question}

    Context:{context}

Include the document source if relevant in your answer.

""")

rag_chain = prompt | llm

def get_pinecone_index():
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        print("Missing Pinecone API key or index name")
        return None
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Pinecone connection failed: {e}")
        return None

async def answer_query(query: str, user_role: str):
    embedding = await asyncio.to_thread(embed_model.embed_query, query)

    index = get_pinecone_index()
    if index is None:
        # Fallback to LLM without context
        final_answer = await asyncio.to_thread(rag_chain.invoke, {"question": query, "context": ""})
        return {
            "answer": final_answer.content + "\n\n(Note: No vector store available; responding without document context.)",
            "sources": []
        }

    try:
        results = await asyncio.to_thread(index.query, vector=embedding, top_k=3, include_metadata=True)
        
        filtered_contexts = []
        sources = set()

        for match in results["matches"]:
            metadata = match["metadata"]
            if metadata.get("role") == user_role:
                filtered_contexts.append(metadata.get("text", "") + "\\n")
                sources.add(metadata.get("source"))
        
        if not filtered_contexts:
            return {"answer": "No relevant context found"}
        
        docs_text = "\\n".join(filtered_contexts)

        final_answer = await asyncio.to_thread(rag_chain.invoke, {"question": query, "context": docs_text})

        return {
            "answer": final_answer.content,
            "sources": list(sources)
        }
    except Exception as e:
        print(f"Pinecone query failed: {e}")
        # Fallback
        final_answer = await asyncio.to_thread(rag_chain.invoke, {"question": query, "context": ""})
        return {
            "answer": final_answer.content + "\n\n(Note: Vector store query failed; responding without document context.)",
            "sources": []
        }
