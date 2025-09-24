import ollama
import qdrant_client
from qdrant_client.models import SearchRequest, QuantizationConfig, ScalarQuantization, HnswConfigDiff, VectorParams
from rich.console import Console
from rich.markdown import Markdown
import os

# --- Configuration ---
# Use environment variables for flexibility. It's better than hardcoding.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "medical_textbook_rag"
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-8B:Q8_0"
CHAT_MODEL = "gpt-oss:120b"

class RAGChatbot:
    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.console = Console()
        # --- Bug Fix ---
        # The original code had a bug where the 'user_question' variable from the
        # global scope was used inside the function instead of the 'question' parameter.
        # This class structure naturally prevents such errors.

    def _embed_question(self, question: str) -> list[float] | None:
        """Embeds the user's question using the specified Ollama model."""
        self.console.print("\n[bold blue]1. Embedding your question...[/bold blue]")
        try:
            response = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=question
            )
            return response['embedding']
        except Exception as e:
            self.console.print(f"[bold red]Error embedding question with Ollama: {e}[/bold red]")
            self.console.print(f"[bold yellow]Is the Ollama server running and the model '{EMBEDDING_MODEL}' available?[/bold yellow]")
            return None

    def _retrieve_context(self, question_embedding: list[float]) -> tuple[str, list[str]] | None:
        """Retrieves relevant context from Qdrant using an optimized search request."""
        self.console.print("[bold blue]2. Retrieving relevant context from Qdrant...[/bold blue]")
        try:
            # --- Expert Refactor: Advanced Search Request ---
            # Instead of a basic client.search(), we use a SearchRequest object.
            # This allows us to specify advanced parameters like 'hnsw_ef'.
            # 'hnsw_ef': Size of the dynamic list for the HNSW graph search.
            # Higher values mean better accuracy but slightly slower search.
            # It's a crucial parameter to tune for optimal performance.
            # Since you are using a GPU, you can afford a higher value like 256 or 512.
            search_request = SearchRequest(
                vector=question_embedding,
                limit=5,
                with_payload=True,
                params={"hnsw_ef": 102400} 
            )
            
            search_results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=search_request.vector,
                limit=search_request.limit,
                search_params=search_request.params
            )

            context = ""
            sources = []
            for result in search_results:
                chunk_text = result.payload.get('text', 'No text content in payload.')
                context += chunk_text + "\n---\n"
                source_info = result.payload.get('section_title', f"ID: {result.id}")
                sources.append(source_info)
            
            source_str = ", ".join(map(str, sources))
            self.console.print(f"[bold green]✔ Retrieved context from sources:[/bold green] {source_str}")
            return context, sources

        except Exception as e:
            self.console.print(f"[bold red]Error searching Qdrant: {e}[/bold red]")
            self.console.print(f"[bold yellow]Is the Qdrant server running and the collection '{COLLECTION_NAME}' accessible?[/bold yellow]")
            return None, None

    def _generate_answer(self, question: str, context: str):
        """Generates an answer using the LLM based on the provided context."""
        prompt_template = f"""
You are an expert cardiologist assistant. Your purpose is to answer questions accurately based ONLY on the provided context.
Do not use any external knowledge. answer to user like answering from your knowledge. If the answer is not contained within the provided context, state that clearly.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:
"""
        self.console.print("[bold blue]3. Generating answer...[/bold blue]\n")
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=[{'role': 'user', 'content': prompt_template}],
            stream=True,
        )
        
        self.console.rule(f"[bold magenta]{CHAT_MODEL} Assistant[/bold magenta]")
        full_response = ""
        for chunk in stream:
            part = chunk['message']['content']
            print(part, end='', flush=True)
            full_response += part
        print() # for newline after streaming
        self.console.rule()

    def ask(self, question: str):
        """Main method to orchestrate the RAG pipeline."""
        question_embedding = self._embed_question(question)
        if not question_embedding:
            return

        context, sources = self._retrieve_context(question_embedding)
        if context is None:
            return

        self._generate_answer(question, context)

if __name__ == "__main__":
    try:
        qdrant_client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Ping the server to ensure connection
        qdrant_client.get_collections() 
        console = Console()
        console.print(f"\n[bold green]✅ Connected to Qdrant instance at {QDRANT_HOST}:{QDRANT_PORT}[/bold green]")

        chatbot = RAGChatbot(qdrant_client)
        
        console.print(f"[bold magenta]🎉 Cardiology RAG Chatbot is ready! (Chat: {CHAT_MODEL} / Embed: {EMBEDDING_MODEL}) 🎉[/bold magenta]")
        console.print("Type your question and press Enter. Type 'exit' to quit.")

        while True:
            user_question = input("\n> ")
            if user_question.lower() == 'exit':
                break
            chatbot.ask(user_question)

    except Exception as e:
        console = Console()
        console.print(f"[bold red]Failed to connect to Qdrant: {e}[/bold red]")
        console.print("[bold yellow]Please ensure the Qdrant Docker container is running.[/bold yellow]")