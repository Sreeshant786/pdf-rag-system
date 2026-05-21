from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunks(question, chunks):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(chunks + [question])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    most_similar_index = similarity.argmax()

    return chunks[most_similar_index]


def generate_answer(question, relevant_chunk):

    return relevant_chunk