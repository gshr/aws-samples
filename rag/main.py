import PyPDF2
from uuid import uuid4
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import os
import streamlit as st


class PDFRag:
    def __init__(
        self, name: str = "1_pdf.pdf", index_name: str = "langchain-test-index"
    ):
        self.doc_name = name
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.QA_PROMPT = PromptTemplate(
            template="""You are an AI assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, respond with: "I don't know."

Context:
{context}

Question: {question}""",
            input_variables=["context", "question"],
        )

        self.embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        self.llm = BedrockLLM(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.1, "max_tokens_to_sample": 2048},
        )
        self.index = self.pc.Index(index_name)
        self.vector_store = PineconeVectorStore(
            index=self.index, embedding=self.embeddings
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_PROMPT},
        )

    def __extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)

    def __extract_chunks(self):
        pdf_text = self.__extract_text_from_pdf(self.doc_name)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        return text_splitter.split_text(pdf_text)

    def store_embeddings(self):
        chunks = self.__extract_chunks()
        documents = [
            Document(page_content=chunk, metadata={"source": self.doc_name})
            for chunk in chunks
        ]
        self.vector_store.add_documents(documents=documents)

    def get_qa_response(self, question: str) -> dict:
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": list(
                {doc.metadata["source"] for doc in result["source_documents"]}
            ),
        }


if __name__ == "__main__":
    rag = PDFRag()
    text = "how to register for a diploma course"
    print(rag.get_qa_response(text))
