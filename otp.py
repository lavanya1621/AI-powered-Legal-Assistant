import streamlit as st
import os
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import fitz  # PyMuPDF for PDF processing
import random
import string
import time
import requests

# Load environment variables
load_dotenv()
# Set page config
st.set_page_config(
    page_title="LexBot - Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

class OTPHandler:
    """Handles OTP generation and delivery through various methods."""
    
    def __init__(self):
        """Initialize the OTP handler."""
        # For email OTP delivery
        self.email_password = os.environ.get("EMAIL_APP_PASSWORD")
        self.sender_email = os.environ.get("SENDER_EMAIL")
        self.admin_email = os.environ.get("ADMIN_EMAIL")
        
        # Determine which delivery method is available
        self.delivery_method = "none"
        
        if self.email_password and self.sender_email and self.admin_email:
            self.delivery_method = "email"
        
        # If no delivery method is available, we'll fall back to displaying on screen
        if self.delivery_method == "none":
            st.warning("No OTP delivery method configured. OTPs will be displayed on screen for demo purposes only.")
            st.info("For production use, set EMAIL_APP_PASSWORD, SENDER_EMAIL, and ADMIN_EMAIL in your .env file.")
    
    def generate_otp(self, length=6):
        """Generate a random numeric OTP."""
        return ''.join(random.choices(string.digits, k=length))
    
    def send_otp(self, otp):
        """Send the OTP via the available delivery method."""
        if self.delivery_method == "email":
            return self.send_otp_via_email(otp)
        else:
            # Fallback to on-screen display for development/demo
            st.warning(f"⚠️ SECURITY NOTICE: In a production environment, the OTP would be sent privately.")
            st.info(f"📱 Demo OTP: {otp}")
            return True, otp
    
    def send_otp_via_email(self, otp):
        """Send the OTP via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = self.admin_email
            message["Subject"] = "LexBot Security Verification Code"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2>LexBot Security Verification</h2>
                <p>Your verification code is:</p>
                <h1 style="font-size: 36px; background-color: #f0f0f0; padding: 10px; text-align: center;">{otp}</h1>
                <p>This code will expire in 5 minutes.</p>
                <p>If you did not request this code, please ignore this email.</p>
            </body>
            </html>
            """
            
            message.attach(MIMEText(body, "html"))
            
            # Connect to Gmail SMTP server
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.email_password)
                server.send_message(message)
            
            st.success(f"OTP sent to {self.admin_email}")
            return True, "OTP sent successfully"
            
        except Exception as e:
            st.error(f"Error sending email: {e}")
            return False, str(e)


class VectorDBHandler:
    """Handles the legal textbook vectorization and retrieval."""
    
    def __init__(self):
        """Initialize the vector database from the textbook."""
        self.is_available = False
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            st.error("No OpenAI API key found in environment variables. Please add OPENAI_API_KEY to your .env file.")
            return
            
        try:
            # Check if the textbook exists
            textbook_path = "textbook.pdf"
            if not os.path.exists(textbook_path):
                st.error(f"Legal textbook not found at {textbook_path}")
                return
                
            # Load textbook if not already in session state
            if 'vector_db' not in st.session_state:
                with st.spinner("Loading legal knowledge base..."):
                    # Load the textbook content
                    text_content = self._extract_text_from_pdf(textbook_path)
                    
                    # Split the content into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text_content)
                    
                    # Create embeddings and vector store using batch processing
                    embeddings = OpenAIEmbeddings(api_key=self.api_key)
                    self._create_vector_db_in_batches(chunks, embeddings)
                    st.success("Legal knowledge base loaded successfully!")
            
            self.is_available = True
            
        except Exception as e:
            st.error(f"Error initializing vector database: {e}")
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file."""
        try:
            text = ""
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()
                
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _create_vector_db_in_batches(self, chunks, embeddings, batch_size=250):
        """Create vector database in batches to avoid token limits."""
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            st.warning("No text chunks found in the document.")
            return
            
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Process the first batch to create the initial vector store
        first_batch_size = min(batch_size, total_chunks)
        progress_text.text(f"Processing batch 1/{(total_chunks + batch_size - 1) // batch_size}: {first_batch_size} chunks")
        
        st.session_state.vector_db = FAISS.from_texts(chunks[:first_batch_size], embeddings)
        
        # Process remaining batches
        for i in range(batch_size, total_chunks, batch_size):
            batch_num = (i // batch_size) + 1
            end_idx = min(i + batch_size, total_chunks)
            current_batch_size = end_idx - i
            
            progress_text.text(f"Processing batch {batch_num}/{(total_chunks + batch_size - 1) // batch_size}: {current_batch_size} chunks")
            progress_bar.progress(i / total_chunks)
            
            # Create a temporary vector store for the current batch
            current_batch = chunks[i:end_idx]
            if current_batch:  # Make sure we have chunks to process
                try:
                    temp_db = FAISS.from_texts(current_batch, embeddings)
                    # Merge with the main vector store
                    st.session_state.vector_db.merge_from(temp_db)
                except Exception as e:
                    st.error(f"Error processing batch {batch_num}: {e}")
        
        progress_bar.progress(1.0)
        progress_text.text("Completed processing all text chunks!")
    
    def query_legal_knowledge(self, question, top_k=3):
        """Query the vector database for relevant legal information."""
        if not self.is_available:
            return "Vector database not available. Please set your OPENAI_API_KEY in .env file."
        
        try:
            results = st.session_state.vector_db.similarity_search(question, k=top_k)
            context_texts = [doc.page_content for doc in results]
            return "\n\n".join(context_texts)
        except Exception as e:
            st.error(f"Error querying vector database: {e}")
            return ""

class GeminiHandler:
    """Handles interactions with Gemini API for direct document processing and querying."""
    
    def __init__(self):
        """Initialize with Google API key from environment."""
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            st.error("No Google API key found in environment variables. Please add GOOGLE_API_KEY to your .env file.")
            self.is_available = False
            return
            
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            # Set up the model - using Gemini 1.5 Pro for best document handling
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            self.is_available = True
        except Exception as e:
            st.error(f"Error initializing Gemini API: {e}")
            self.is_available = False
    
    def process_document_query(self, file_path, question, legal_context=""):
        """Process document directly with Gemini's multimodal capabilities."""
        if not self.is_available:
            return "Gemini API not available. Please set your GOOGLE_API_KEY in .env file."
            
        try:
            # Read the file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Determine file type
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                mime_type = "application/pdf"
            elif file_extension in ['.docx', '.doc']:
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                return f"Unsupported file format: {file_extension}. Please upload a PDF or DOCX file."
            
            # Create a detailed prompt for document analysis with legal context
            prompt = f"""You are LexBot, an expert assistant specializing in Indian legal and business documents.

DOCUMENT ANALYSIS INSTRUCTIONS:
1. First, identify the type of document (e.g., GST invoice, contract, agreement, policy) by examining headers, titles and content.
2. Review the entire document thoroughly before responding.
3. Focus on extracting precise information related to the question.
4. Pay special attention to names, dates, amounts, terms, and legal clauses.

USER QUESTION:
{question}

{"ADDITIONAL LEGAL CONTEXT:" if legal_context else ""}
{legal_context}

RESPONSE GUIDELINES:
- Provide a direct, concise answer to the question.
- Quote specific relevant parts of the document when appropriate.
- If specific information is not found in the document, state that clearly.
- For legal documents, note any disclaimers but avoid giving legal advice.
- For financial documents like invoices, clearly state important financial figures, dates, parties involved, and tax details.
- Format monetary amounts, dates, and percentages consistently.
- If the document is incomplete or unclear, mention this in your response.
- When applicable, use the additional legal context to enrich your analysis but focus primarily on the uploaded document.
"""
            
            # Use multipart request to send document and prompt
            with st.spinner("LexBot is analyzing your document..."):
                response = self.model.generate_content(
                    contents=[
                        {
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": mime_type, "data": file_data}}
                            ]
                        }
                    ],
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 1024,
                    }
                )
            
            return response.text.strip()
            
        except Exception as e:
            st.error(f"Error processing document with Gemini: {e}")
            return f"Error processing document: {str(e)}"


class LegalDocumentAssistant:
    """Main class for document processing and question answering."""
    
    def __init__(self):
        self.gemini = GeminiHandler()
        self.vector_db = VectorDBHandler()
        self.otp_handler = OTPHandler()
        self.file_path = None
        self.document_info = None
    
    def process_document(self, uploaded_file):
        """Save uploaded document for querying."""
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            self.file_path = tmp_file.name
        
        # Return basic document info
        self.document_info = {
            "document_id": uploaded_file.name,
            "file_path": self.file_path,
            "file_size": uploaded_file.size,
            "file_type": os.path.splitext(uploaded_file.name)[1]
        }
        
        return self.document_info
    
    def generate_and_send_otp(self):
        """Generate an OTP and send it via the configured method."""
        otp = self.otp_handler.generate_otp()
        success, message = self.otp_handler.send_otp(otp)
        
        if success:
            # Store OTP in session state with a timestamp
            st.session_state.current_otp = {
                "code": otp,
                "timestamp": time.time(),
                "verified": False
            }
            return True
        else:
            st.error(f"Failed to send OTP: {message}")
            return False
    
    def verify_otp(self, entered_otp):
        """Verify the entered OTP against the stored one."""
        if "current_otp" not in st.session_state:
            return False, "No OTP was generated. Please request a new OTP."
        
        stored_otp = st.session_state.current_otp
        
        # Check if OTP is expired (5 minutes validity)
        if time.time() - stored_otp["timestamp"] > 300:
            return False, "OTP has expired. Please request a new OTP."
        
        # Check if OTP matches
        if entered_otp == stored_otp["code"]:
            st.session_state.current_otp["verified"] = True
            return True, "OTP verified successfully!"
        else:
            return False, "Invalid OTP. Please try again."
    
    def ask_question(self, question):
        """Ask a question about the currently loaded document."""
        if not self.file_path:
            return {"answer": "No document has been uploaded yet."}
        
        # First, query the legal textbook for relevant context
        legal_context = ""
        if self.vector_db.is_available:
            with st.spinner("Retrieving relevant legal information..."):
                legal_context = self.vector_db.query_legal_knowledge(question)
        
        # Then process the document with the additional context
        answer = self.gemini.process_document_query(self.file_path, question, legal_context)
        
        return {
            "answer": answer,
            "document": self.document_info["document_id"] if self.document_info else "Unknown"
        }


# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = LegalDocumentAssistant()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 'document_info' not in st.session_state:
    st.session_state.document_info = None
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'answer_ready' not in st.session_state:
    st.session_state.answer_ready = False
if 'answer_content' not in st.session_state:
    st.session_state.answer_content = None


def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        border-radius: 10px;
        margin-top: 20px;
        padding: 15px;
    }
    .user-message {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: #333333;
    }
    .bot-message {
        background-color: #E6F3FF;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: #333333;
    }
    .info-box {
        background-color: #F7F9FB;
        border-left: 4px solid #2E86C1;
        padding: 10px;
        margin-bottom: 15px;
        color: #333333;
    }
    .otp-container {
        background-color: #F8F9F9;
        border-radius: 10px;
        border: 1px solid #D5DBDB;
        padding: 15px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 class='main-header'>⚖️ LexBot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Your Legal Document Assistant</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload your legal document", type=["pdf", "docx"])
        
        if uploaded_file is not None and (not st.session_state.document_uploaded or 
                                         st.session_state.document_info is None or 
                                         st.session_state.document_info.get("document_id") != uploaded_file.name):
            st.info("Processing document...")
            st.session_state.document_uploaded = True
            st.session_state.document_info = st.session_state.assistant.process_document(uploaded_file)
            st.session_state.chat_history = []  # Clear chat history for new document
            st.success(f"Document uploaded successfully!")
            
        if st.session_state.document_uploaded:
            st.write("---")
            st.subheader("Document Information")
            if st.session_state.document_info:
                st.write(f"📄 **File:** {st.session_state.document_info.get('document_id', 'Unknown')}")
                st.write(f"📦 **Size:** {round(st.session_state.document_info.get('file_size', 0) / 1024, 2)} KB")
                st.write(f"📑 **Type:** {st.session_state.document_info.get('file_type', 'Unknown').upper()[1:]}")
            
            if st.button("Clear Document"):
                # Clean up temporary file
                if st.session_state.assistant.file_path and os.path.exists(st.session_state.assistant.file_path):
                    try:
                        os.unlink(st.session_state.assistant.file_path)
                    except:
                        pass
                
                st.session_state.document_uploaded = False
                st.session_state.document_info = None
                st.session_state.chat_history = []
                st.session_state.assistant.file_path = None
                st.session_state.pending_question = None
                st.session_state.answer_ready = False
                st.session_state.answer_content = None
                st.rerun()
        
        st.write("---")
        st.markdown("### About LexBot")
        st.info("""
        LexBot helps you understand legal documents by answering your questions.
        
        It can process:
        - PDF documents (including scanned PDFs)
        - Word documents (DOCX)
        
        LexBot combines document analysis with a legal knowledge base to provide more accurate answers.
        
        **Security Feature**: Verification required before viewing answers.
        """)
        
        st.caption("Powered by Google Gemini & OpenAI Embeddings")
            
    # Main area - Chat interface
    if not st.session_state.document_uploaded:
        st.info("Please upload a document in the sidebar to start chatting with LexBot.")
        
        # Show demo information
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### How it works")
            st.markdown("""
            1. **Upload** your legal document
            2. **Ask questions** about the document
            3. **Verify with code** sent to your email or displayed on screen
            4. **Get answers** based on document content and legal knowledge
            """)
            
        with col2:
            st.markdown("### Example questions")
            st.markdown("""
            - What is the term of this lease agreement?
            - Who are the parties in this contract?
            - What are my obligations under this agreement?
            - Is there a confidentiality clause?
            - What's the GST amount in this invoice?
            """)
            
    else:
        # Display chat history
        st.subheader("Chat with your document")
        
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"<div class='user-message'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-message'><b>LexBot:</b> {message['content']}</div>", unsafe_allow_html=True)
        
        # OTP Verification Section - Show when there's a pending question
        if st.session_state.pending_question is not None and not st.session_state.answer_ready:
            st.markdown("<div class='otp-container'>", unsafe_allow_html=True)
            st.subheader("🔒 Security Verification Required")
            st.info("To view the answer to your question, verification is required.")
            
            # OTP request and verification
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Send Verification Code"):
                    if st.session_state.assistant.generate_and_send_otp():
                        # No need for success message as it's shown in the OTP handler
                        pass
                    else:
                        st.error("Failed to send verification code. Please try again.")
            
            with col2:
                with st.form(key="otp_form", clear_on_submit=True):
                    otp_input = st.text_input("Enter Verification Code", placeholder="123456", max_chars=6)
                    verify_button = st.form_submit_button("Verify Code")
                    
                    if verify_button and otp_input:
                        success, message = st.session_state.assistant.verify_otp(otp_input)
                        if success:
                            st.session_state.answer_ready = True
                            # Add assistant's response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": st.session_state.answer_content["answer"]
                            })
                            # Clear pending question and answer content after adding to chat history
                            st.session_state.pending_question = None
                            st.session_state.answer_content = None
                            st.success("Verification successful! Answer displayed.")
                            st.rerun()
                        else:
                            st.error(message)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        st.write("---")
        
        # Use a form to prevent resubmission on page refresh
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_input("Ask a question about your document:", placeholder="e.g., What is the total amount in this invoice?")
            submit_button = st.form_submit_button("Ask LexBot")
            
            if submit_button and question:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Get answer from assistant but don't display yet
                response = st.session_state.assistant.ask_question(question)
                
                # Store the question and answer for later display after OTP verification
                st.session_state.pending_question = question
                st.session_state.answer_content = response
                st.session_state.answer_ready = False
                
                # Force a rerun to update the chat UI and show OTP verification
                st.rerun()


if __name__ == "__main__":
    main()