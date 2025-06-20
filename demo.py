#!/usr/bin/env python3
"""
Demo script for the Document Q&A System.
Shows how to use the system programmatically.
"""

import os
from pathlib import Path
from src.qa_chain import DocumentQAChain
from src.utils import setup_environment

def main():
    """
    Demo the Document Q&A system with sample usage.
    """
    print("🚀 Document Q&A System Demo")
    print("=" * 50)
    
    try:
        # Setup environment
        print("📋 Setting up environment...")
        setup_environment()
        print("✅ Environment configured")
        
        # Initialize QA chain
        print("\n🤖 Initializing QA chain...")
        qa_chain = DocumentQAChain(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500
        )
        print("✅ QA chain initialized")
        
        # Check for documents
        documents_path = "./documents"
        if not any(Path(documents_path).iterdir()):
            print(f"\n⚠️  No documents found in {documents_path}")
            print("Please add some PDF, DOCX, or TXT files to the documents/ directory")
            return
        
        # Load documents
        print(f"\n📚 Loading documents from {documents_path}...")
        result = qa_chain.load_documents(documents_path)
        
        if result["status"] == "error":
            print(f"❌ Error loading documents: {result['error']}")
            return
            
        print(f"✅ Loaded {result['documents_loaded']} documents")
        print(f"📊 Created {result['document_stats']['total_documents']} chunks")
        
        # Demo questions
        demo_questions = [
            "What are the main topics covered in the documents?",
            "Can you provide a summary of the key points?",
            "Who are the important people or organizations mentioned?",
            "What are the main recommendations or conclusions?"
        ]
        
        print("\n" + "=" * 50)
        print("🤔 Demo Questions & Answers")
        print("=" * 50)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 40)
            
            try:
                response = qa_chain.ask(question)
                print(f"💡 Answer: {response['answer']}")
                
                if response.get('sources'):
                    print(f"\n📚 Sources ({len(response['sources'])} documents):")
                    for j, source in enumerate(response['sources'][:2], 1):
                        print(f"   {j}. {source['filename']} - {source['content_preview'][:100]}...")
                        
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
            print("\n" + "-" * 50)
        
        # System stats
        print("\n📊 System Statistics:")
        stats = qa_chain.get_system_stats()
        for key, value in stats.items():
            if key != "vector_store_stats":
                print(f"   {key}: {value}")
        
        print("\n✅ Demo completed successfully!")
        print("\n🌐 To run the web interface: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("\n💡 Make sure you have:")
        print("   - Set your OPENAI_API_KEY environment variable")
        print("   - Installed all requirements: pip install -r requirements.txt")
        print("   - Added documents to the documents/ directory")

if __name__ == "__main__":
    main()
