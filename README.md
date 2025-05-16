# LexBot – AI-Powered Legal Document Assistant

LexBot is an AI-powered legal assistant that helps users upload legal documents (PDF or Word), ask questions about them, and get detailed answers—secured with OTP-based verification. It leverages **Google Gemini Pro**, **OpenAI embeddings**, and a **legal vector knowledge base** for enhanced legal insight.

---

## 🚀 Features

- 📄 Upload **PDF** and **DOCX** legal documents
- ❓ Ask natural-language questions about the content
- 🔒 OTP-based **security verification** before revealing answers
- 🧠 Combines document understanding with **legal textbook embeddings**
- 🧬 Powered by **Gemini Pro** and **LangChain + FAISS**
- 📥 Document type detection, legal clause extraction, and context-aware answers
- 🌐 Streamlit UI with dynamic OTP display or email delivery

---

## 🧠 How It Works

```mermaid
flowchart TD
    U[User] -->|Uploads Document| A[Streamlit App]
    A --> B[Process with Gemini Pro]
    A --> C[Query VectorDB (Legal Textbook)]
    B --> D[Answer Generated]
    C --> B
    D --> E{OTP Verification}
    E -->|Valid OTP| F[Display Answer]
    E -->|Invalid/Expired OTP| G[Error Message]
    A --> H[Send OTP via Email or Show On-Screen]
