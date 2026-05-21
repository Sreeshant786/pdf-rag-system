import streamlit as st

from pdf_processing import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from vector_store import (
    create_vector_store,
    retrieve_relevant_chunks
)

from llm_handler import generate_answer


st.title("PDF Question Answering System")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file is not None:

    st.success("PDF Uploaded Successfully!")

    # Extract text
    text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    st.write("Total Chunks:", len(chunks))

    # Create vector store
    index, embeddings = create_vector_store(chunks)

    st.success("Vector Store Created Successfully!")

    # User question
    question = st.text_input(
        "Ask a Question from PDF"
    )

    if question:

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            question,
            index,
            chunks
        )

        # Generate AI answer
        answer = generate_answer(
            question,
            relevant_chunks
        )

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Retrieved Context")

        for chunk in relevant_chunks:

            st.write(chunk)

            st.write("-------------------")