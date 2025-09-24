import ollama
import qdrant_client
import logging
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client.http import models

# --- 1. Configuration ---
CONFIG = {
    "ollama_model": 'dengcao/Qwen3-Embedding-8B:Q8_0',
    "input_document_path": Path("UPTODATE_text_only.md"),
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "collection_name": "uptodate_V1",
    "embedding_batch_size": 10, # Smaller batch size is better for individual calls
    "hnsw_m": 32,
    "hnsw_ef_construct": 200,
}

# --- 2. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def chunk_by_section(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads a structured Markdown file and groups it into semantic chunks
    based on custom page separators and top-level headings.
    """
    if not file_path.exists():
        logging.error(f"File not found at {file_path}")
        return []

    logging.info(f"Loading document from {file_path}...")
    content = file_path.read_text(encoding="utf-8")

    page_contents = re.split(r'--- Page \d+ ---', content)
    page_numbers = re.findall(r'--- Page (\d+) ---', content)
    all_chunks = []

    logging.info("Splitting document into chunks based on pages and headings...")
    for i, page_text in enumerate(page_contents[1:]):
        page_num = int(page_numbers[i])
        blocks = page_text.strip().split('\n\n')

        current_heading = ""
        current_paragraphs = []

        def create_chunk(heading, paragraphs, p_num):
            """Helper function to avoid code repetition."""
            if not paragraphs and not heading.strip('# '):
                return
            doc_text = "\n\n".join([heading] + paragraphs).strip()
            all_chunks.append({
                "document": doc_text,
                "payload": {
                    "page_number": p_num,
                    "section_title": heading.lstrip('# ').strip()
                }
            })

        for block in blocks:
            if block.strip().startswith('#'):
                create_chunk(current_heading, current_paragraphs, page_num)
                current_heading = block.strip()
                current_paragraphs = []

            elif block.strip().startswith('--- [TABLE DETECTED] ---'):
                create_chunk(current_heading, current_paragraphs, page_num)
                current_heading = ""
                current_paragraphs = []
                
                all_chunks.append({
                        "document": block.strip(),
                        "payload": {
                            "page_number": page_num,
                            "section_title": "Table Data"
                        }
                    })
            else:
                current_paragraphs.append(block.strip())

        create_chunk(current_heading, current_paragraphs, page_num)

    return all_chunks


def main():
    """
    Main function to run the data processing pipeline with original methods.
    """
    # --- 1. Chunk the Document ---
    logging.info(f"Starting the chunking process for: {CONFIG['input_document_path']}")
    document_chunks = chunk_by_section(CONFIG['input_document_path'])

    if not document_chunks:
        logging.error("No chunks were created. Exiting.")
        return
    logging.info(f"✅ Created {len(document_chunks)} semantic chunks.")

    # --- 2. Initialize Clients ---
    try:
        logging.info(f"Initializing Qdrant client at {CONFIG['qdrant_host']}:{CONFIG['qdrant_port']}")
        qdrant = qdrant_client.QdrantClient(host=CONFIG['qdrant_host'], port=CONFIG['qdrant_port'])
        
        logging.info(f"Pinging Ollama and determining vector size with model: '{CONFIG['ollama_model']}'")
        embedding_result = ollama.embeddings(model=CONFIG['ollama_model'], prompt="ping")
        vector_size = len(embedding_result['embedding'])
        logging.info(f"  - Detected vector size: {vector_size}")

    except Exception as e:
        logging.error(f"❌ Failed to connect to Qdrant or Ollama. Please ensure services are running. Error: {e}")
        return

    # --- 3. Create Qdrant Collection ---
    collection_name = CONFIG['collection_name']
    try:
        qdrant.get_collection(collection_name=collection_name)
        logging.info(f"Collection '{collection_name}' already exists. Upserting data.")
    except Exception:
        logging.info(f"Collection '{collection_name}' not found. Creating a new one with optimized settings.")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE, on_disk=True),
            hnsw_config=models.HnswConfigDiff(m=CONFIG['hnsw_m'], ef_construct=CONFIG['hnsw_ef_construct'])
        )
        logging.info("Creating payload indexes for 'section_title' and 'page_number'.")
        qdrant.create_payload_index(collection_name=collection_name, field_name="section_title", field_schema="keyword")
        qdrant.create_payload_index(collection_name=collection_name, field_name="page_number", field_schema="integer")

    # --- 4. Generate Embeddings and Store in Qdrant ---
    logging.info("Embedding chunks and uploading to Qdrant...")
    batch_size = CONFIG['embedding_batch_size']
    
    for i in range(0, len(document_chunks), batch_size):
        batch_chunks = document_chunks[i:i+batch_size]
        texts_to_embed = [chunk['document'] for chunk in batch_chunks]
        logging.info(f"  - Processing batch {i//batch_size + 1} ({len(texts_to_embed)} chunks)...")

        try:
            # Reverted to original, one-by-one embedding call as requested
            embeddings = [
                ollama.embeddings(model=CONFIG['ollama_model'], prompt=text)['embedding']
                for text in texts_to_embed
            ]

            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={**chunk['payload'], "text": chunk['document']}
                )
                for chunk, embedding in zip(batch_chunks, embeddings)
            ]

            qdrant.upsert(collection_name=collection_name, points=points, wait=True)

        except Exception as e:
            logging.error(f"❌ An error occurred during batch {i//batch_size + 1}: {e}")
            continue

    # --- 5. Verification ---
    count_result = qdrant.count(collection_name=collection_name, exact=True)
    logging.info(f"\n🎉 Indexing complete!")
    logging.info(f"✅ Successfully stored {count_result.count} vectors in the '{collection_name}' collection.")


if __name__ == "__main__":
    main()