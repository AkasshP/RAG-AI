import streamlit as st                                    # UI
from PyPDF2 import PdfReader                              # PDF → text
from langchain.text_splitter import RecursiveCharacterTextSplitter  # chunking
from langchain_community.vectorstores import FAISS         # FAISS via community package
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings    # Embeddings provider
from langchain.llms import HuggingFacePipeline             # Local HF LLM provider
from transformers import pipeline                         # HuggingFace pipeline helper# Chat LLM provider
from dotenv import load_dotenv                            # .env support
load_dotenv()  # Set OPENAI_API_KEY in a .env file at project root

# ——— Helpers —————————————————————————————————————————————————————————————

def get_pdf_text(pdf_files):
    """Extract all text from a list of UploadedFile-like objects."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def get_text_chunks(text, chunk_size=5000, overlap=500):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)

def get_vector_store(chunks, embeddings) -> FAISS:
    """Build (and save) a FAISS index from text chunks."""
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")
    return db

def load_vector_store(embeddings) -> FAISS:
    """Reload the previously-saved FAISS index."""
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

def get_qa_chain(llm, prompt_template) -> object:
    """Create a simple QA chain with your prompt."""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# ——— Main UI / Flow ——————————————————————————————————————————————————————

def main():
    st.set_page_config(page_title="Chat PDF RAG Demo")
    st.header("🔍 Ask Questions of Your PDFs")

    # Sidebar: upload + processing
    with st.sidebar:
        st.title("Upload & Process")
        pdf_docs = st.file_uploader(
            "Upload one or more PDFs",
            type="pdf",
            accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF first.")
                return

            with st.spinner("Extracting text…"):
                raw_text = get_pdf_text(pdf_docs)

            with st.spinner("Splitting into chunks…"):
                chunks = get_text_chunks(raw_text)

            # build embeddings & index
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            with st.spinner("Building FAISS index…"):
                get_vector_store(chunks, embeddings)

            st.success("Finished processing. You can now ask questions!")

    # Main: question box + answer
    question = st.text_input("Enter your question here:")
    if question:
        # Reload our index & set up chain
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = load_vector_store(embeddings)
        docs = db.similarity_search(question, k=5)

        # create chain
        prompt_template = (
            "Answer the question as fully as possible using ONLY the provided context. "
            "If the answer cannot be found in the context, reply “Answer not available in the context.”\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        hf_pipe = pipeline(
                "text-generation",
                model="gpt2",            # you can pick another like "distilgpt2" or "EleutherAI/gpt-neo-125M"
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                device=-1                
                )
        llm = HuggingFacePipeline(pipeline=hf_pipe)
        chain = get_qa_chain(llm, prompt_template)

        # run and display
        with st.spinner("Thinking…"):
            result = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
        st.subheader("Answer")
        st.write(result["output_text"])

if __name__ == "__main__":
    main()
