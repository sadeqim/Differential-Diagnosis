import ollama
import qdrant_client
from rich.console import Console
from rich.markdown import Markdown

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_textbook_rag" 
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-8B:Q8_0" 
CHAT_MODEL = "gpt-oss:120b"                         
console = Console()

def query_rag_chatbot(client, question):
    """
    Queries the existing RAG system and streams the answer.
    """
    console.print("\n[bold blue]1. Embedding your question...[/bold blue]")
    try:

        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=user_question
        )

        question_embedding = response['embedding']

    except Exception as e:
        console.print(f"[bold red]Error embedding question with Ollama: {e}[/bold red]")
        console.print(f"[bold yellow]Is the Ollama server running and the model '{EMBEDDING_MODEL}' available?[/bold yellow]")
        return

    console.print("[bold blue]2. Retrieving relevant context from Qdrant...[/bold blue]")
    try:
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=3 
        )
    except Exception as e:
        console.print(f"[bold red]Error searching Qdrant: {e}[/bold red]")
        console.print(f"[bold yellow]Is the Qdrant server running and the collection '{COLLECTION_NAME}' accessible?[/bold yellow]")
        return
        
    context = ""
    sources = []
    for result in search_results:
        chunk_text = result.payload.get('text', 'No text content in payload.')
        context += chunk_text + "\n---\n"
        
        source_info = result.payload.get('section_title', f"ID: {result.id}")
        sources.append(source_info)
    
    source_str = ", ".join(map(str, sources))
    console.print(f"[bold green]✔ Retrieved context from sources:[/bold green] {source_str}")

    prompt_template = f"""
You are an expert cardiologist assistant. Your purpose is to answer questions accurately based ONLY on the provided context.
Do not use any external knowledge. If the answer is not contained within the provided context, state that clearly.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:
"""

    console.print("[bold blue]3. Generating answer...[/bold blue]\n")
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[{'role': 'user', 'content': prompt_template}],
        stream=True,
    )
    
    console.rule(f"[bold magenta]{CHAT_MODEL} Assistant[/bold magenta]")
    for chunk in stream:
        part = chunk['message']['content']
        print(part, end='', flush=True)
    console.rule()

if __name__ == "__main__":
    qdrant_client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    console.print("\n[bold green]✅ Connected to existing Qdrant instance.[/bold green]")
    console.print(f"[bold magenta]🎉 Cardiology RAG Chatbot is ready! (Chat: {CHAT_MODEL} / Embed: {EMBEDDING_MODEL}) 🎉[/bold magenta]")
    console.print("Type your question and press Enter. Type 'exit' to quit.")

    while True:
        user_question = input("\n> ")
        if user_question.lower() == 'exit':
            break
        query_rag_chatbot(qdrant_client, user_question)