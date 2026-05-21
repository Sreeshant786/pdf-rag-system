from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small"
)

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Answer the question briefly and accurately using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    result = generator(
        prompt,
        max_new_tokens=100
    )

    return result[0]["generated_text"]