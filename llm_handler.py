from transformers import pipeline

# Smaller lightweight model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an AI assistant.

    Answer the question briefly and accurately using only the provided context.

     If the answer is not found, say:
    "Answer not found in document."

    Context:
    {context}

    Question:
    {question}

     Short Answer:
    """

    result = generator(
        prompt,
        max_length=100
    )

    return result[0]["generated_text"]