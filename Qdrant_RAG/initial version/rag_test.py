import ollama
import qdrant_client
import argparse


OLLAMA_EMBEDDING_MODEL = 'dengcao/Qwen3-Embedding-8B:Q8_0'
OLLAMA_LLM_MODEL = 'qwen3:30b-a3b-fp16'
COLLECTION_NAME = 'medical_textbook_rag'
NUM_RESULTS_TO_RETRIEVE = 10


def main():
    parser = argparse.ArgumentParser(description="Query a document using a RAG system with Qdrant.")
    parser.add_argument("question", type=str, help="The question you want to ask the document.")
    args = parser.parse_args()
    user_question = args.question

    print(f"❓ User Question: {user_question}\n")

    try:
        client = qdrant_client.QdrantClient(host="localhost", port=6333)
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error connecting to Qdrant or finding collection: {e}")
        print(f"Please ensure your Qdrant container is running and you have run the 'index_document.py' script successfully to create the '{COLLECTION_NAME}' collection.")
        return

    print("🧠 Embedding the question...")
    try:
        response = ollama.embeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            prompt=user_question
        )
        query_embedding = response['embedding']
    except Exception as e:
        print(f"❌ Error communicating with Ollama for embedding: {e}")
        return

    print("🔍 Retrieving relevant context from the database...")
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=NUM_RESULTS_TO_RETRIEVE
    )

    print("📝 Building the prompt with retrieved context...")
    context_str = ""
    for i, result in enumerate(search_results):
        payload = result.payload
        context_str += f"Source {i+1} (Page {payload.get('page_number', 'N/A')}, Section: {payload.get('section_title', 'N/A')}):\n"
        context_str += payload.get('text', 'No text available.') + "\n\n"

    prompt_template = f"""
You are a helpful assistant who answers questions based ONLY on the provided context.
Do not use any outside knowledge. If the answer is not in the context, say "The answer is not available in the provided context."
Cite the page number and section title from the most relevant source when you answer.

CONTEXT:
---
{context_str}
---

QUESTION:
{user_question}

ANSWER:
"""

    print(f"💬 Sending prompt to '{OLLAMA_LLM_MODEL}' to generate the final answer...")
    try:
        llm_response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_template,
                },
            ]
        )
        final_answer = llm_response['message']['content']
    except Exception as e:
        print(f"❌ Error communicating with Ollama for generation: {e}")
        return

    print("\n" + "="*50)
    print("✅ Generated Answer:")
    print("="*50)
    print(final_answer)
    print("\n" + "="*50)
    print("📚 Sources Used:")
    print("="*50)
    print(context_str)


if __name__ == "__main__":
    main()
