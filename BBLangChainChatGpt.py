import pdfplumber
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Load the PDF and extract its content
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Convert PDF text into Document format
def text_to_documents(text):
    # Split text into chunks, as LLMs might have a token limit
    chunks = text.split('\n\n')  # Split by double newline
    return [Document(page_content=chunk) for chunk in chunks]


# Convert the extracted text into Ollama embeddings
def get_embeddings_from_text(text_chunks):
    embedding_model = OllamaEmbeddings(model="llama3")  # Ensure "llama3" is available locally
    return embedding_model.embed_documents([chunk.page_content for chunk in text_chunks])


# Save embeddings to FAISS vector store
def save_to_faiss_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, OllamaEmbeddings(model="llama3"))
    vector_store.save_local("faiss_index")  # Save the index locally for future queries
    return vector_store

# Load the FAISS index if already saved
def load_faiss_index():
    return FAISS.load_local("faiss_index", OllamaEmbeddings(model="llama3"))

# Build a Q&A chain with FAISS and Ollama
def build_qa_engine(vector_store):
    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())
    return qa_chain

# 5. Retrieve relevant chunks to augment the query with context
def retrieve_context(vector_store, query, top_k=3):
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)[:top_k]
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context
# Ask a question using the Q&A engine
def ask_question(qa_chain, query):
    response = qa_chain.run(query)
    return response

# 6. Ask a question with additional context from relevant chunks
def ask_with_context(qa_chain, vector_store, query):
    context = retrieve_context(vector_store, query)
    enriched_query = f"Context:\n{context}\n\nQuestion: {query}"
    response = qa_chain.run(enriched_query)
    return response

def process_pdf_for_qa(pdf_path):
    # Step 1: Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    # Step 2: Convert text to LangChain Document format
    documents = text_to_documents(extracted_text)

    # Step 3: Generate embeddings and save to FAISS
    embeddings = get_embeddings_from_text(documents)
    vector_store = save_to_faiss_vector_store(documents, embeddings)

    # Step 4: Build Q&A engine
    qa_engine = build_qa_engine(vector_store)
    return qa_engine

# Load the FAISS index and ask a question
def ask_from_saved_index(query):
    vector_store = load_faiss_index()
    qa_engine = build_qa_engine(vector_store)
    return ask_question(qa_engine, query)

# Usage example:
pdf_path = "/Users/bharatbonde/PycharmProjects/CyberMetric/content/data/o1-system-card.pdf"
qa_engine = process_pdf_for_qa(pdf_path)

# Example query
#query = "What is fairness and bias evaluations?"
#query = "How does o1 compare to gpt-4o in disallowed content evaluations?"
query = "o1 and gpt-4o are open AI models.How does o1 compare to gpt-4o in disallowed content evaluations?"
response = ask_question(qa_engine, query)
print(response)