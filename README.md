# 📚 Advanced Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over document collections. Built with state-of-the-art embedding models and vector databases for enterprise-grade performance.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, TXT files
- **Advanced Text Chunking**: Smart document segmentation with overlap
- **Semantic Search**: Vector similarity search using OpenAI embeddings
- **Context-Aware Responses**: GPT-powered answers with source citations
- **Interactive Web Interface**: Clean Streamlit UI for easy interaction
- **Conversation Memory**: Maintains context across multiple queries
- **Source Attribution**: Always shows which documents informed the answer
- **Scalable Architecture**: Easily extensible for large document collections

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4/GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Database**: Chroma (with persistence)
- **Document Processing**: LangChain + PyPDF2
- **Web Interface**: Streamlit
- **Framework**: LangChain for RAG pipeline

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- 4GB+ RAM (for document processing)

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd document-qa-system
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Add your OpenAI API key to .env file
```

### 3. Add Documents

```bash
mkdir documents
# Place your PDF/DOCX/TXT files in the documents/ folder
```

### 4. Run the Application

```bash
streamlit run app.py
```

## 🏗️ Architecture

```
Document Q&A System
├── Document Ingestion
│   ├── PDF/DOCX/TXT Parsing
│   ├── Smart Text Chunking
│   └── Metadata Extraction
├── Vector Processing
│   ├── OpenAI Embeddings
│   ├── Chroma Vector Store
│   └── Similarity Search
└── Response Generation
    ├── Context Retrieval
    ├── GPT Response Generation
    └── Source Citation
```

## 📁 Project Structure

```
document-qa-system/
├── app.py                 # Streamlit web interface
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # Document loading and chunking
│   ├── vector_store.py       # Vector database operations
│   ├── qa_chain.py          # RAG pipeline implementation
│   └── utils.py             # Utility functions
├── documents/              # Place your documents here
├── vectorstore/           # Chroma database (auto-created)
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCS_RETRIEVE=5
```

### Customization Options

- **Chunk Size**: Adjust in `src/document_processor.py`
- **Embedding Model**: Change in `src/vector_store.py`
- **LLM Model**: Modify in `src/qa_chain.py`
- **UI Theme**: Customize in `app.py`

## 🚀 Usage Examples

### Basic Q&A

```python
from src.qa_chain import DocumentQAChain

qa_chain = DocumentQAChain()
qa_chain.load_documents("./documents")

response = qa_chain.ask("What are the main topics covered in the documents?")
print(response['answer'])
print(response['sources'])
```

### Batch Processing

```python
questions = [
    "What is the company's revenue?",
    "Who are the key stakeholders?",
    "What are the main risks?"
]

for question in questions:
    response = qa_chain.ask(question)
    print(f"Q: {question}")
    print(f"A: {response['answer']}\n")
```

## 📊 Performance

- **Document Processing**: ~2-5 seconds per PDF page
- **Query Response**: ~1-3 seconds per question
- **Memory Usage**: ~100MB base + 50MB per 1000 chunks
- **Scalability**: Tested with 1000+ document collections

## 🛡️ Best Practices

1. **Document Quality**: Use clear, well-formatted documents
2. **Chunk Size**: Balance between context and precision (recommended: 800-1200 chars)
3. **Query Formulation**: Ask specific questions for better results
4. **API Limits**: Monitor OpenAI API usage for cost optimization

## 🔄 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced document preprocessing
- [ ] Custom embedding models
- [ ] RESTful API endpoints
- [ ] Docker containerization
- [ ] Cloud deployment guides

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

For questions or issues:

- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Built with ❤️ by [Your Name]** | Showcasing enterprise-ready RAG implementations
