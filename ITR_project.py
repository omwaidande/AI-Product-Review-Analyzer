'''
Load
chunk
embed
retrieve
generate
get_embeddings, chunking, cosine_similarity, load_document, answer_question
'''
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# get embeddings
def get_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings.tolist()


# cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2)


# chunking
def chunk_text(text, chunk_size=400, overlap=50):
    param = []
    for p in text.split('\n\n'):
        if p.strip():
            param.append(p.strip())
    if len(param) >= 3:
        return param
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start = start + chunk_size - overlap
    return chunks


# loading document
def load_document(filepath):
    if not os.path.exists(filepath):
        print(f'Error: file {filepath} not found')
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    if not text.strip():
        print('Error: document is empty')
        return None
    chunks = chunk_text(text)
    print(f'Loaded {len(chunks)} chunks')
    stored = []
    for c in chunks:
        # get_embeddings accepts a single string here, encode() handles it fine
        stored.append({'text': c, 'embedding': get_embeddings(c)})
    return stored


# answer question
def answer_question(question, stored):
    q_embedding = get_embeddings(question)

    scores = []
    for s in stored:
        score = cosine_similarity(q_embedding, s['embedding'])
        scores.append((score, s['text']))  # was missing outer tuple parens (bug)

    scores.sort(reverse=True)  # highest similarity first
    top_chunks = [text for score, text in scores[:2]]
    context = '\n\n'.join(top_chunks)

    prompt = f'''you are a product review analyst,
                using ONLY the customer reviews below, answer the question,
                look for patterns across multiple reviews rather than just
                quoting one, mention roughly how common the pattern seems,
                if the reviews don't cover this topic, say so clearly,
                keep the answer short - 2-4 sentences max,
Answer the question USING ONLY the below context.
If the answer is not there, say
"I don't have the information in document."

context: {context}
question: {question}
'''

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # any Groq-hosted chat model works here
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers strictly from the given context."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


# simple CLI runner tying load -> chunk -> embed -> retrieve -> generate together
if __name__ == "__main__":
    filepath = input("Path to document: ").strip()
    stored_chunks = load_document(filepath)

    if stored_chunks is None:
        raise SystemExit(1)

    while True:
        question = input("\nAsk a question (or 'quit' to exit): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = answer_question(question, stored_chunks)
        print(f"\nAnswer: {answer}")