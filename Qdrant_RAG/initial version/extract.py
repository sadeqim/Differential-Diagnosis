import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.base_models import InputFormat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_log = logging.getLogger(__name__)

def custom_render_to_markdown(document) -> str:

    markdown_parts = []
    current_page = -1

    text_map = {item.self_ref: item for item in document.texts}
    table_map = {item.self_ref: item for item in document.tables}
    item_map = {**text_map, **table_map}

    for ref_item in document.body.children:
        item = item_map.get(ref_item.cref)
        if not item:
            continue

        if hasattr(item, 'prov') and item.prov:
            page_no = item.prov[0].page_no
            if page_no != current_page:
                current_page = page_no
                markdown_parts.append(f"--- Page {current_page + 1} ---\n\n")
        
        item_class_name = type(item).__name__

        if item_class_name == 'SectionHeaderItem':
            level = getattr(item, 'level', 2)
            markdown_parts.append(f"{'#' * level} {item.text}\n\n")
        
        elif item_class_name == 'TextItem':
            markdown_parts.append(f"{item.text}\n\n")

        elif item_class_name == 'TableItem':
            markdown_parts.append("--- [TABLE DETECTED] ---\n")
            table_text_parts = [" | ".join([cell.text.strip() for cell in row]) for row in item.data.grid]
            markdown_parts.append("\n".join(table_text_parts))
            markdown_parts.append("\n--- [END OF TABLE] ---\n\n")
    
    return "".join(markdown_parts)

def convert_pdf_to_markdown(pdf_path: Path, markdown_path: Path):
    """Main conversion function."""
    try:
        if not pdf_path.exists():
            _log.error(f"Input file not found: {pdf_path}")
            return

        _log.info("Initializing DocumentConverter...")
        doc_converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline)
        })

        _log.info(f"Starting conversion for '{pdf_path.name}'...")
        results_generator = doc_converter.convert_all([pdf_path])

        for result in results_generator:
            if result.document:
                _log.info("Applying custom rendering with page numbers...")
                markdown_content = custom_render_to_markdown(result.document)
                markdown_path.write_text(markdown_content, encoding='utf-8')
                _log.info(f"✅ Successfully converted and saved to '{markdown_path}'")
            else:
                _log.error(f"Failed to process document: {result.input.file.name}")
            break
            
    except Exception as e:
        _log.critical(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    input_pdf = Path("./UPTODATE.pdf") 
    output_md = Path(f"{input_pdf.stem}_text_only.md")
    convert_pdf_to_markdown(input_pdf, output_md)