import PyPDF2
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import os
import streamlit as st


class PDFRag:
    def __init__(self, name: str = "1_pdf.pdf", index_name: str = "langchain-test-index"):
        self.doc_name = name
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        self.embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        self.llm = BedrockLLM(
            model_id="anthropic.claude-v2",
            model_kwargs={
                "temperature": 0.1,
                "max_tokens_to_sample": 2048
            }
        )
        
        # Initialize Pinecone index
        self.index = self.pc.Index(index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def __extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(
                page.extract_text() 
                for page in pdf_reader.pages
            )

    def __extract_chunks(self):
        pdf_text = self.__extract_text_from_pdf(self.doc_name)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_text(pdf_text)

    def store_embeddings(self):
        chunks = self.__extract_chunks()
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": self.doc_name}
            ) for chunk in chunks
        ]
        self.vector_store.add_documents(documents=documents)

    def get_qa_response(self, question: str) -> dict:
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": list({
                doc.metadata["source"] 
                for doc in result["source_documents"]
            })
        }


def main():
    st.set_page_config(
        page_title="PDF AI Assistant",
        page_icon="📄",
        layout="centered"
    )
    
    st.markdown("""
    <style>
    .stTextInput input {
        border-radius: 20px;
        padding: 15px;
    }
    .stButton button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border-radius: 15px;
        padding: 10px 24px;
    }
    .stMarkdown {
        padding: 15px;
        border-radius: 10px;
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("📂 Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
                with st.spinner("Analyzing documents..."):
                    for uploaded_file in uploaded_files:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        rag = PDFRag(name=uploaded_file.name)
                        rag.store_embeddings()
                        os.remove(uploaded_file.name)
                st.success("Documents processed successfully!")

    st.title("📄 PDF AI Assistant")
    st.caption("Ask anything about your uploaded documents")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        rag = PDFRag()
        response = rag.get_qa_response(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            with st.expander("View Sources"):
                if response["sources"]:
                    for source in response["sources"]:
                        st.markdown(f"📄 {source}")
                else:
                    st.warning("No sources found")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"]
        })

if __name__ == "__main__":
    main()
