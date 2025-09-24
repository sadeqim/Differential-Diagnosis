# rag_core.py
import os
from ollama import AsyncClient as AsyncOllamaClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import HnswConfigDiff 
from typing import AsyncGenerator, List, Dict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "uptodate_V1"
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-8B:Q8_0"
CHAT_MODEL = "qwen3:32b" # "symptoma/medgemma3:27b" #"gpt-oss:120b" 

class RAGChatbot:

    def __init__(self, ollama_client: AsyncOllamaClient, qdrant_client: AsyncQdrantClient):
        self.ollama_client = ollama_client
        self.qdrant_client = qdrant_client
        self.embedding_model = EMBEDDING_MODEL
        self.chat_model = CHAT_MODEL
        self.collection_name = COLLECTION_NAME

    async def _embed_question(self, question: str) -> List[float]:
        response = await self.ollama_client.embeddings(model=self.embedding_model, prompt=question)
        return response['embedding']

    async def _retrieve_context(self, question_embedding: List[float]) -> tuple[str, List[str]]:

        search_results = await self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=question_embedding,
            limit=5,
            with_payload=True,
            search_params={"hnsw_ef": 2048000}
        )
        
        context_parts = [result.payload.get('text', '') for result in search_results]
        sources = sorted(list(set(result.payload.get('section_title', f"ID: {result.id}") for result in search_results)))
        return "\n---\n".join(context_parts), sources

    async def _generate_response_stream(self, question: str, context: str) -> AsyncGenerator[str, None]:
        # prompt = f"""
        # You are a specialized AI assistant with expertise in clinical cardiology, designed to assist medical professionals.
        # Your primary function is to synthesize and present information exclusively from the provided CONTEXT to answer the user's QUESTION.

        # **CRITICAL INSTRUCTIONS:**
        # 1.  **Strictly Adhere to Context:** You MUST derive your entire answer from the CONTEXT below. Do not use any external knowledge. If the answer is not in the context, explicitly state that the provided materials do not contain the necessary information.
        # 2.  **Structure and Formatting:**
        #     * **Detailed Breakdown:** Begin with a comprehensive answer. If the information involves categories, criteria, or lists (e.g., symptoms, diagnostic criteria, treatment options), you **MUST** format this information as a Markdown table for maximum clarity. The table should have clear, descriptive headers.
        #     * **Emphasis:** Use bold markdown (`**text**`) to highlight all key clinical terms, symptoms, drug names, and other critical concepts.
        #     * **Concluding Summary:** After the detailed breakdown, provide a final, concise summary of the key findings. This summary **MUST** be in a bulleted list format.
        # 3.  **Tone and Style:** Maintain a professional, objective, and clinical tone. Your response should be clear, direct, and evidence-based (from the context).

        # ---
        # CONTEXT:
        # {context}
        # ---

        # QUESTION: {question}

        # ---
        # ANSWER:
        # """
        prompt = f"""
            You are an expert cardiologist assistant. Your purpose is to answer questions accurately based ONLY on the provided context.
            Do not use any external knowledge. answer to user like answering from your knowledge. If the answer is not contained within the provided context, state that clearly.

            CONTEXT:
            ---
            {context}
            ---

            QUESTION: {question}

            ANSWER:
            """
        stream = await self.ollama_client.chat(model=self.chat_model, messages=[{'role': 'user', 'content': prompt}], stream=True)
        async for chunk in stream:
            if content := chunk['message']['content']:
                yield content

    async def stream_rag_response(self, question: str) -> AsyncGenerator[Dict, None]:

        try:
            question_embedding = await self._embed_question(question)
            logging.info(f"Question Embdded")
            context, sources = await self._retrieve_context(question_embedding)
            logging.info(f"Context Retrived")

            
            
            if sources:
                yield {"type": "sources", "data": sources}
            
            if not context:
                yield {"type": "llm_chunk", "data": "I could not find any relevant information in my knowledge base to answer that question."}
                return

            answer_stream = self._generate_response_stream(question, context)

            
            async for text_chunk in answer_stream:
                yield {"type": "llm_chunk", "data": text_chunk}
            logging.info(f"Response Sent")
        except Exception as e:
            error_message = f"Critical error in RAG pipeline: {e}"
            print(f"ERROR: {error_message}")
            yield {"type": "error", "data": error_message}