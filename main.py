from app.document_loader import load_pdf_documents
from pathlib import Path


def main():
    pdf_paths = {pdf_path.stem: pdf_path for pdf_path in sorted(Path("data").glob("*.pdf"))}
    documents = load_pdf_documents(pdf_paths)
    print(f"Loaded {len(documents)} document pages from data/")

    for document in documents[:3]:
        print(
            f"- {document.metadata['name']} [page {document.metadata['page']}] "
            f"{document.page_content[:80]!r}"
        )


if __name__ == "__main__":
    main()
