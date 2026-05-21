from transformers import pipeline

generator = pipeline(
    "text2text-generation",
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
        max_length=200,
        do_sample=False
    )

    return response[0]["generated_text"]