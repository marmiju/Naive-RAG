import chromadb
from chromadb.utils import embedding_functions

sentence_embedder = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path="./chroma_db/store")
collection = client.get_or_create_collection(
    name="developer_knowledge"
)

def create_embeddings(chunks: list[str]) -> list[list[float]]:
    print("Embedding started...")

    embeddings = sentence_embedder(chunks)

    print("Embedding completed!")
    return embeddings

def store_embeded_cunks(embeded_cunks,chunks):
    collection.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeded_cunks,
    )
    data = collection.get(
    limit=2,
    include=["documents", "embeddings"],
    )

    


