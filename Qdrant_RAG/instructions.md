**instructions**

To create the vector db (using chroma db), run these scripts in order:
1. table_extract.py (change the pdf name and path in the code)
2. chunk_embed_store.py
3. rag_test.py (to run: python rag_test.py "inital_test_prompt")

docker run --gpus=all -p 6333:6333 -p 6334:6334 -e QDRANT__GPU__INDEXING=1 -d  qdrant/qdrant:gpu-nvidia-latest

qdrant/qdrant                   gpu-nvidia-latest              ddce4193c758   8 weeks ago    843MB

  python - <<'PY'
  from pathlib import Path
  from pypdf import PdfMerger

  src_dir = Path("/home/user01/DD/docminer_pro/pdfs")
  output_pdf = Path("merged_output.pdf")

  merger = PdfMerger()
  for pdf in sorted(src_dir.glob("*.pdf")):
      merger.append(str(pdf))
  merger.write(output_pdf)
  merger.close()
  PY