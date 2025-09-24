# main.py
import uvicorn
import json
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Literal

from rag_core import RAGChatbot
from ollama import AsyncClient as AsyncOllamaClient
from qdrant_client import AsyncQdrantClient

app_state = {}
RAG_MODEL_ID = "rag/uptodate-expert" 

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Server starting up...")
    from rag_core import QDRANT_HOST, QDRANT_PORT
    qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ollama_client = AsyncOllamaClient()
    app_state["chatbot"] = RAGChatbot(ollama_client=ollama_client, qdrant_client=qdrant_client)
    print("INFO:     Startup complete. Application is ready.")
    yield
    await app_state["chatbot"].qdrant_client.close()
    print("INFO:     Qdrant client closed. Server shutting down.")

app = FastAPI(lifespan=lifespan, title="Cardiology RAG API", version="3.0.0")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True

class ModelInfo(BaseModel):
    id: str = Field(..., description="The model identifier.")
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "user"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

@app.get("/", summary="Root endpoint for health check")
def root():
    return {"message": "Cardiology RAG API is running."}

@app.get("/v1/models", response_model=ModelList, summary="List available models")
async def list_models():
    """Provides a list of available models to Open WebUI."""
    model_data = [ModelInfo(id=RAG_MODEL_ID)]
    return ModelList(data=model_data)

@app.post("/v1/chat/completions", summary="OpenAI-compatible chat endpoint")
async def chat_endpoint(request: OllamaChatRequest):
    """Handles chat requests using the standard OpenAI streaming format."""
    chatbot = app_state.get("chatbot")
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot is not available")

    if request.model != RAG_MODEL_ID:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model}' not found. This endpoint only serves '{RAG_MODEL_ID}'."
        )

    async def response_generator():
        stream_id = f"chatcmpl-{uuid.uuid4()}"
        
        first_chunk_data = {"id": stream_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        yield f"data: {json.dumps(first_chunk_data)}\n\n"

        user_question = request.messages[-1].content
        
        rag_stream = chatbot.stream_rag_response(user_question)
        
        async for chunk in rag_stream:
            chunk_type = chunk.get("type")
            chunk_data = chunk.get("data", "")
            
            content_to_stream = ""
            if chunk_type == "sources":
                content_to_stream = "**Sources:**\n" + "\n".join(f"- {s}" for s in chunk_data) + "\n\n---\n\n"
            elif chunk_type in ("llm_chunk", "error"):
                content_to_stream = chunk_data
            
            if content_to_stream:
                response_delta = {"id": stream_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": content_to_stream}, "finish_reason": None}]}
                yield f"data: {json.dumps(response_delta)}\n\n"

        final_chunk_data = {"id": stream_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final_chunk_data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)