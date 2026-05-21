from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_vector_store(chunks):

    # Convert chunks into embeddings
    embeddings = embedding_model.encode(chunks)

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    index.add(embeddings)

    return index, embeddings


def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=2
):

    # Convert question into embedding
    question_embedding = embedding_model.encode([question])

    # Search similar chunks
    distances, indices = index.search(
        np.array(question_embedding),
        top_k
    )

    # Retrieve matching chunks
    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks