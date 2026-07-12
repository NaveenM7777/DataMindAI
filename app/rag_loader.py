import fitz  # PyMuPDF
import streamlit as st


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file (Streamlit UploadedFile object).

    Returns:
        full_text: str — all extracted text, concatenated across pages
        page_texts: list of str — text per page (index 0 = page 1)
    """

    try:

        # Read the uploaded file's bytes and open with PyMuPDF
        pdf_bytes = uploaded_file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        page_texts = []

        for page in doc:

            text = page.get_text()

            page_texts.append(text)

        doc.close()

        full_text = "\n\n".join(page_texts)

        return full_text, page_texts

    except Exception as e:

        st.error(f"❌ Failed to extract text from PDF: {e}")

        return None, None


def get_pdf_metadata(uploaded_file):
    """
    Returns basic metadata about the PDF: page count, file size.
    Resets the file pointer afterward so extract_text_from_pdf can
    still read it fresh if called separately.
    """

    try:

        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        metadata = {
            "page_count": doc.page_count,
            "file_size_kb": round(len(pdf_bytes) / 1024, 2)
        }

        doc.close()

        uploaded_file.seek(0)

        return metadata

    except Exception as e:

        st.error(f"❌ Failed to read PDF metadata: {e}")

        return None