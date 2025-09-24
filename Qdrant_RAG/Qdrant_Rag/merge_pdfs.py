"""Utility to merge all PDFs in a folder into a single PDF."""
from pathlib import Path
from pypdf import PdfReader, PdfWriter


def merge_pdfs(source_dir: Path, output_path: Path) -> None:
    writer = PdfWriter()
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {source_dir}")

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        writer.write(file_obj)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge all PDFs in a folder into one PDF file.")
    parser.add_argument(
        "source",
        type=Path,
        help="Folder containing the PDFs to merge",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("merged_output.pdf"),
        help="Path for the merged PDF (defaults to ./merged_output.pdf)",
    )

    args = parser.parse_args()
    merge_pdfs(args.source, args.output)

    print(f"Merged PDF saved to {args.output.resolve()}")
