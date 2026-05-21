from transformers import pipeline

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small"
)

def generate_answer(question, context_chunks):

    context = " ".join(context_chunks)

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = generator(
        prompt,
        max_new_tokens=100
    )

    return response[0]["generated_text"]