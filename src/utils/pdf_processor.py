"""
PDF Processing Utility for EduNotes
Extracts text content from PDF files for note generation
"""
import PyPDF2
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    """Handles PDF file processing and text extraction"""

    def __init__(self):
        self.logger = logger

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                self.logger.info(f"Extracting text from PDF with {num_pages} pages")

                # Extract text from all pages
                text_content = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)

                full_text = "\n\n".join(text_content)

                if not full_text.strip():
                    self.logger.warning("No text extracted from PDF")
                    return None

                self.logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
                return full_text

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return None

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """
        Extract text from PDF bytes (for uploaded files).

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            from io import BytesIO

            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            self.logger.info(f"Extracting text from uploaded PDF with {num_pages} pages")

            # Extract text from all pages
            text_content = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)

            full_text = "\n\n".join(text_content)

            if not full_text.strip():
                self.logger.warning("No text extracted from uploaded PDF")
                return None

            self.logger.info(f"Successfully extracted {len(full_text)} characters from uploaded PDF")
            return full_text

        except Exception as e:
            self.logger.error(f"Error extracting text from uploaded PDF: {e}")
            return None

    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'title': None,
                    'author': None,
                    'subject': None,
                    'creator': None
                }

                if pdf_reader.metadata:
                    metadata['title'] = pdf_reader.metadata.get('/Title', None)
                    metadata['author'] = pdf_reader.metadata.get('/Author', None)
                    metadata['subject'] = pdf_reader.metadata.get('/Subject', None)
                    metadata['creator'] = pdf_reader.metadata.get('/Creator', None)

                return metadata

        except Exception as e:
            self.logger.error(f"Error extracting PDF metadata: {e}")
            return {'num_pages': 0}


# Singleton instance
_pdf_processor = None


def get_pdf_processor() -> PDFProcessor:
    """Get or create PDF processor instance"""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessor()
    return _pdf_processor
