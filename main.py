import os
import fitz  # PyMuPDF
import docx
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

class DocumentProcessor:
    """Handles document loading and text extraction."""
    
    def extract_text(self, file_path):
        """Extract text from PDF or DOCX files."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF using PyMuPDF."""
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX using python-docx."""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text


class SimpleVectorDB:
    """A simple vector database implementation for document retrieval."""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_texts(self, texts):
        """Add texts to the vector database."""
        if not texts:
            return
            
        self.chunks.extend(texts)
        embeddings = self.embedding_model.encode(texts)
        self.embeddings.extend(embeddings)
    
    def similarity_search(self, query, k=5):
        """Find the most similar chunks to the query."""
        if not self.chunks:
            return []
            
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get indices of top k most similar chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return the most similar chunks
        return [{"content": self.chunks[i], "similarity": similarities[i]} for i in top_indices]


class DocumentIndexer:
    """Indexes document content for efficient retrieval."""
    
    def __init__(self):
        # Smaller chunks for more precise retrieval
        self.chunk_size = 500  # Increased for better context
        self.chunk_overlap = 100
        self.db = SimpleVectorDB()
    
    def create_chunks(self, text):
        """Split text into overlapping chunks."""
        if not text or len(text.strip()) == 0:
            return []
            
        # Split by sections and paragraphs
        sections = []
        
        # Try to identify document sections using numbered patterns like "1. PROPERTY DESCRIPTION"
        section_matches = re.finditer(r'(\d+\.\s+[A-Z\s]+)\n', text)
        section_positions = [(m.start(), m.group(1)) for m in section_matches]
        
        if section_positions:
            # Add document header as first section if not included
            if section_positions[0][0] > 0:
                sections.append(text[:section_positions[0][0]])
                
            # Add each section
            for i in range(len(section_positions)):
                start = section_positions[i][0]
                end = section_positions[i+1][0] if i+1 < len(section_positions) else len(text)
                sections.append(text[start:end])
        else:
            # Fallback to paragraph-based chunking
            paragraphs = re.split(r'\n\s*\n', text)
            sections = paragraphs
        
        # Further chunk any long sections
        chunks = []
        for section in sections:
            if len(section.split()) <= self.chunk_size:
                chunks.append(section)
            else:
                words = section.split()
                i = 0
                while i < len(words):
                    end = min(i + self.chunk_size, len(words))
                    chunk = " ".join(words[i:end])
                    chunks.append(chunk)
                    i += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def index_document(self, text):
        """Split the document into chunks and index them."""
        chunks = self.create_chunks(text)
        self.db.add_texts(chunks)
        return self.db


class GeminiHandler:
    """Handles interactions with Gemini API for direct document processing and querying."""
    
    def __init__(self):
        """Initialize with Google API key from environment."""
        load_dotenv()  # Load environment variables
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            print("Warning: No Google API key found in environment variables.")
            self.is_available = False
            return
            
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            # Set up the model
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            self.is_available = True
            print("Gemini API initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            self.is_available = False
    
    def direct_pdf_query(self, file_path, question):
        """Process PDF directly with Gemini's multimodal capabilities for direct querying."""
        if not self.is_available:
            return "Gemini API not available. Please set your GOOGLE_API_KEY in .env file."
            
        try:
            # Read the PDF file
            with open(file_path, 'rb') as f:
                pdf_data = f.read()
            
            # Generate a prompt for the PDF analysis
            prompt = f"""You are a helpful assistant analyzing a legal document.
            
Please answer the following question accurately and concisely based on the provided document:
{question}

If the information is not clearly provided in the document, state that explicitly.
Base your answer solely on the provided document."""

            # Use multipart request to send PDF and prompt - only one API call
            response = self.model.generate_content(
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "application/pdf", "data": pdf_data}}
                        ]
                    }
                ],
                generation_config={"temperature": 0.2, "max_output_tokens": 1024}
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error processing PDF with Gemini: {e}")
            return f"Error processing document: {str(e)}"
    
    def answer_with_context(self, question, context_chunks, document_type="Legal Document"):
        """Answer questions using retrieved context chunks."""
        if not self.is_available:
            return "Gemini API not available. Please set your GOOGLE_API_KEY in .env file."
        
        if not context_chunks:
            return "No relevant information found in the document to answer this question."
        
        try:
            # Format context from retrieved chunks
            context = "\n\n".join([chunk["content"] for chunk in context_chunks[:3]])  # Limiting to top 3 chunks
            
            # Prepare the prompt for Gemini
            prompt = f"""You are an expert assistant for legal documents.
            
Document Type: {document_type}

Below are relevant sections from the document:
{context}

Given this information, please answer the following question accurately and concisely:
{question}

If the information is not provided in the document, state that explicitly."""
            
            # Call Gemini API - single API call with context
            response = self.model.generate_content(
                contents=[prompt],
                generation_config={"temperature": 0.2, "max_output_tokens": 1024}
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return f"Error processing your question: {str(e)}"


class LegalDocumentAssistant:
    """Main class for document processing and question answering."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.indexer = DocumentIndexer()
        self.gemini = GeminiHandler()
        self.vectordb = None
        self.document_text = ""
        self.file_path = None
        self.is_ocr_mode = False
    
    def process_document(self, file_path):
        """Process a document and prepare it for querying."""
        print(f"Processing document: {file_path}")
        self.file_path = file_path
        
        # Extract text from the document
        self.document_text = self.processor.extract_text(file_path)
        
        # Check if we got meaningful text or need OCR mode
        if not self.document_text or len(self.document_text.strip()) < 100:
            print("Minimal text extracted. Switching to direct OCR mode.")
            self.is_ocr_mode = True
            return {
                "document_id": os.path.basename(file_path),
                "mode": "Direct OCR/Gemini Mode",
                "text_length": len(self.document_text) if self.document_text else 0
            }
        
        # Use RAG mode - index the document
        self.is_ocr_mode = False
        self.vectordb = self.indexer.index_document(self.document_text)
        
        return {
            "document_id": os.path.basename(file_path),
            "mode": "RAG Mode",
            "text_length": len(self.document_text)
        }
    
    def ask_question(self, question):
        """Ask a question about the currently loaded document."""
        if not self.file_path:
            raise ValueError("No document has been processed yet.")
        
        print(f"Question: {question}")
        
        # OCR Mode: Use direct PDF processing with Gemini
        if self.is_ocr_mode:
            if not self.file_path.lower().endswith('.pdf'):
                return {"answer": "Direct processing only works with PDF files in OCR mode."}
                
            answer = self.gemini.direct_pdf_query(self.file_path, question)
            return {
                "answer": answer,
                "mode": "Direct OCR/Gemini Mode",
                "source_chunks": ["Direct PDF processing"]
            }
        
        # RAG Mode: Use vector search and then query with context
        retrieved_chunks = self.vectordb.similarity_search(question)
        answer = self.gemini.answer_with_context(question, retrieved_chunks)
        
        return {
            "answer": answer,
            "mode": "RAG Mode",
            "source_chunks": [chunk["content"][:100] + "..." for chunk in retrieved_chunks[:3]]
        }


def main():
    """Interactive main function."""
    assistant = LegalDocumentAssistant()
    
    # Get document path from user
    while True:
        file_path = input("Enter the path to your document (PDF or DOCX): ")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        break
    
    # Process the document
    result = assistant.process_document(file_path)
    print(f"Document processed in {result['mode']}")
    
    # Interactive question loop
    print("\nYou can now ask questions about the document.")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            result = assistant.ask_question(question)
            print(f"\nAnswer: {result['answer']}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("Thank you for using the Legal Document Assistant!")


if __name__ == "__main__":
    main()