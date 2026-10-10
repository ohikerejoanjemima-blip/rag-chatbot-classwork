"""PDF parsing and session-local in-memory vector search."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np


MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 500
MAX_TEXT_CHARS = 2_000_000


def extract_pages(data, filename):
    from pypdf import PdfReader

    if len(data) > MAX_PDF_BYTES:
        raise ValueError("Choose a PDF smaller than 25 MB.")

    if not data:
        raise ValueError("The PDF is empty.")

    try:
        reader = PdfReader(BytesIO(data))

        if reader.is_encrypted:
            raise ValueError("Upload an unencrypted PDF.")

        if len(reader.pages) > MAX_PAGES:
            raise ValueError("Choose a PDF with 500 pages or fewer.")

        pages = []
        total = 0

        for number, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            total += len(text)

            if total > MAX_TEXT_CHARS:
                raise ValueError(
                    "This PDF contains too much text. Split it into smaller PDFs."
                )

            if text:
                pages.append(
                    {
                        "text": text,
                        "page": number,
                        "source": Path(filename).name,
                    }
                )

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            "Could not read the PDF. Try an unencrypted, text-based PDF."
        ) from exc

    if not pages:
        raise ValueError(
            "No selectable text was found. Run OCR on the PDF first."
        )

    return pages, len(reader.pages)


@dataclass
class PDFIndex:
    documents: list
    vectors: np.ndarray
    embeddings: object
    page_count: int
    chunk_count: int

    def invoke(self, question):
        """Return the four most relevant PDF chunks."""

        with self.embeddings.lock:
            query_vector = np.array(
                self.embeddings.embeddings.embed_query(question),
                dtype=np.float32,
            )

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return self.documents[:4]

        scores = self.vectors @ query_vector
        scores = scores / (
            np.linalg.norm(self.vectors, axis=1) * query_norm + 1e-10
        )

        best_indices = np.argsort(scores)[::-1][:4]

        return [self.documents[i] for i in best_indices]

    def close(self):
        """Nothing to close because the index lives in memory."""
        pass


def index_pdf(data, filename, embedding_resource):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages, page_count = extract_pages(data, filename)

    docs = [
        Document(
            page_content=p["text"],
            metadata={
                "page": p["page"],
                "source": p["source"],
            },
        )
        for p in pages
    ]

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    ).split_documents(docs)

    if not chunks:
        raise ValueError("No searchable text was found in the PDF.")

    texts = [chunk.page_content for chunk in chunks]

    vectors = []

    with embedding_resource.lock:
        for start in range(0, len(texts), 8):
            batch = texts[start : start + 8]
            batch_vectors = embedding_resource.embeddings.embed_documents(batch)
            vectors.extend(batch_vectors)

    vectors = np.array(vectors, dtype=np.float32)

    return PDFIndex(
        documents=chunks,
        vectors=vectors,
        embeddings=embedding_resource,
        page_count=page_count,
        chunk_count=len(chunks),
    )