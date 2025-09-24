import ollama
import qdrant_client
from qdrant_client.http import models
import re
from pathlib import Path
import uuid
from tqdm.auto import tqdm


OLLAMA_MODEL = 'dengcao/Qwen3-Embedding-8B:Q8_0'
COLLECTION_NAME = 'medical_textbook_rag'
INPUT_DOCUMENT = Path("./merged_output_text_only.md")


def chunk_by_section(file_path: Path) -> list[dict]:
    """
    Loads a structured Markdown file and groups it into semantic chunks.
    (This function is DB-agnostic and remains unchanged)
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return []

    content = file_path.read_text(encoding="utf-8")

    page_contents = re.split(r'--- Page \d+ ---', content)
    page_numbers = re.findall(r'--- Page (\d+) ---', content)
    all_chunks = []

    for i, page_text in enumerate(tqdm(page_contents[1:], desc="Chunking pages", unit="page", leave=False)):
        page_num = int(page_numbers[i])
        blocks = page_text.strip().split('\n\n')

        current_heading = ""
        current_paragraphs = []

        for block in blocks:
            if block.strip().startswith('#'):
                if current_paragraphs:
                    chunk_text = "\n\n".join([current_heading] + current_paragraphs).strip()
                    all_chunks.append({
                        "document": chunk_text,
                        "payload": {
                            "page_number": page_num,
                            "section_title": current_heading.lstrip('# ').strip()
                        }
                    })

                current_heading = block.strip()
                current_paragraphs = []

            elif block.strip().startswith('--- [TABLE DETECTED] ---'):
                 all_chunks.append({
                        "document": block.strip(),
                        "payload": {
                            "page_number": page_num,
                            "section_title": "Table Data"
                        }
                    })

            else:
                current_paragraphs.append(block.strip())

        if current_paragraphs:
            chunk_text = "\n\n".join([current_heading] + current_paragraphs).strip()
            all_chunks.append({
                "document": chunk_text,
                "payload": {
                    "page_number": page_num,
                    "section_title": current_heading.lstrip('# ').strip()
                }
            })

    return all_chunks

def main():
    """
    Main function to run the Qdrant indexing pipeline against a Docker container.
    """
    print(f"Loading and chunking document: {INPUT_DOCUMENT}")
    document_chunks = chunk_by_section(INPUT_DOCUMENT)

    if not document_chunks:
        print("No chunks were created. Exiting.")
        return

    print(f"Created {len(document_chunks)} chunks.")

    print(f"Initializing Qdrant client to connect to Docker container at localhost:6333")
    client = qdrant_client.QdrantClient(host="localhost", port=6333)

    print(f"Determining vector size with Ollama model: '{OLLAMA_MODEL}'")
    try:
        embedding_result = ollama.embeddings(model=OLLAMA_MODEL, prompt="Get vector size")
        vector_size = len(embedding_result['embedding'])
        print(f"  - Detected vector size: {vector_size}")
    except Exception as e:
        print(f"Could not connect to Ollama to get vector size: {e}")
        return

    print(f"Creating Qdrant collection: '{COLLECTION_NAME}'")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"  - Collection created successfully.")


    print(f"Embedding chunks and uploading to Qdrant...")

    batch_size = 10
    for i in tqdm(range(0, len(document_chunks), batch_size), desc="Embedding batches", unit="batch", leave=False):
        batch_chunks = document_chunks[i:i+batch_size]

        texts_to_embed = [chunk['document'] for chunk in batch_chunks]
        payloads = [chunk['payload'] for chunk in batch_chunks]

        print(f"  - Processing batch {i//batch_size + 1} ({len(batch_chunks)} chunks)...")

        try:
            embeddings = [
                ollama.embeddings(model=OLLAMA_MODEL, prompt=text)['embedding']
                for text in texts_to_embed
            ]

            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={**payload, "text": text}
                )
                for embedding, payload, text in zip(embeddings, payloads, texts_to_embed)
            ]

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
        except Exception as e:
            print(f"❌ An error occurred during embedding or storing: {e}")
            print("  - Please ensure your Ollama instance is running and the model is available.")
            return

    count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"Indexing complete!")
    print(f"Successfully stored {count_result.count} chunks in the '{COLLECTION_NAME}' collection.")


if __name__ == "__main__":
    main()
