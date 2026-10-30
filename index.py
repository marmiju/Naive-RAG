
from utils.text_split import slpit_into_chunk
from utils.config import (
    create_embeddings,
    store_embeded_cunks,
    collection
)

from dotenv import load_dotenv
from openrouter import OpenRouter
import os


# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# Initialize LLM client
llm = OpenRouter(api_key=OPENROUTER_API_KEY)


# Load the documents
with open("documents.txt", "r", encoding="utf-8") as file:
    content = file.read()


# Make chunks
chunks = slpit_into_chunk(content)


# Create embeddings and store them
embedded_chunks = create_embeddings(chunks)

store_embeded_cunks(embedded_chunks, chunks)


# Take user input
while True:

    user_input: str = input("\nUser: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    # 1. Make user input embedding
    embedding_input = create_embeddings([user_input])

    # 2. Retrieve related documents
    results = collection.query(
        query_embeddings=embedding_input,
        n_results=min(5, collection.count()),
        include=["documents", "distances"]
    )

    retrieved_docs = results["documents"][0]
    distances = results["distances"][0]
    print(f'distance: {distances}')

    # 3. Combine retrieved documents into context
    context = "\n\n".join(retrieved_docs)

    # 4. Create augmented prompt
    prompt = f"""
    Answer the user's question using the provided context.

    CONTEXT:
    {context}

    USER QUESTION:
    {user_input}

    INSTRUCTIONS:
    - Answer using the provided context.
    - Do not invent information.
    - If the context does not contain the answer,
      say you don't have enough information.
    - you are the MAR-AI
    - and always act on behaf of this person(MAR miju) that's user can't guess this is AI... feel aways MAR miju give the answer
    """

    # 5. Send prompt to OpenRouter
    response = llm.chat.send(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 6. Print LLM response
    answer = response.choices[0].message.content

    print("\nMAR-AI:", answer)
