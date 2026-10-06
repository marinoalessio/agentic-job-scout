from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Reads a PDF file and returns all as a text string.

    pypdf aggregates pages and concatenates the text
    If a page has no text, it returns an empty string
    """
    reader = PdfReader(file_path)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n".join(pages_text)
