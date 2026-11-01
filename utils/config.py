import chromadb
from chromadb.utils import embedding_functions

sentence_embedder = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path="./chroma_db/store")
collection = client.get_or_create_collection(
    name="developer_knowledge"
)

def create_embeddings(chunks: list[str]) -> list[list[float]]:
    print("Embedding started...")

    embeddings = sentence_embedder(chunks )

    print("Embedding completed!")
    return embeddings

def store_embeded_cunks(embeded_cunks,chunks, target_collection=None):
    col = target_collection if target_collection is not None else collection
    existing_ids = col.get()["ids"]
    if existing_ids:
        col.delete(ids=existing_ids)
    col.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeded_cunks,
    )
    data = col.get(
    limit=2,
    include=["documents", "embeddings"],
    )

    


