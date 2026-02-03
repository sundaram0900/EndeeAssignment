import streamlit as st
from endee_client import EndeeClient
from rag_engine import RAGEngine
import os

# Configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")

st.set_page_config(page_title="Endee RAG Demo", layout="wide")

st.title("Endee Vector Database - RAG Demo")
st.markdown("""
This application demonstrates a **Retrieval Augmented Generation (RAG)** system using **Endee** as the high-performance vector database.
""")

# Initialize components
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
    
if 'endee_client' not in st.session_state:
    st.session_state.endee_client = EndeeClient(base_url=ENDEE_URL)

with st.sidebar:
    st.header("1. Ingest Data")
    text_input = st.text_area("Paste text content here:", height=200)
    if st.button("Ingest Content"):
        if text_input:
            with st.spinner("Chunking and Embedding..."):
                chunks = st.session_state.rag_engine.chunk_text(text_input)
                vectors = st.session_state.rag_engine.generate_embeddings(chunks)
                
                # Insert into Endee
                # For demo simplicity, we just use a generic collection/index concept
                # The EndeeClient needs to handle the specifics
                try:
                    response = st.session_state.endee_client.insert_vectors(vectors, chunks)
                    st.success(f"Successfully indexed {len(chunks)} chunks!")
                    if response:
                        st.json(response)
                except Exception as e:
                    st.error(f"Error inserting into Endee: {e}")
        else:
            st.warning("Please paste some text first.")

st.header("2. Semantic Search")
query = st.text_input("Ask a question or search for a topic:")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            # Generate query vector
            query_vector = st.session_state.rag_engine.generate_embeddings([query])[0]
            
            # Search Endee
            try:
                # Assuming search returns list of matches
                # Adjust parsing based on actual Endee API response
                results = st.session_state.endee_client.search(query_vector, k=3)
                
                st.subheader("Results:")
                st.write(results)
                
                # If results structure is known, we can prettify:
                # for res in results.get('matches', []):
                #     st.markdown(f"**Score:** {res['score']}")
                #     st.info(res['document'])
                
            except Exception as e:
                st.error(f"Error searching Endee: {e}")
    else:
        st.warning("Please enter a query.")
