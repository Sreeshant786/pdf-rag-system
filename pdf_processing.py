from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    return text


def split_text_into_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    return chunks