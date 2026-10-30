
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_retrieval(collection, query_embedding, results):

    # -----------------------------------------
    # 1. Get ALL stored document embeddings
    # -----------------------------------------

    total_chunks = collection.count()

    if total_chunks == 0:
        print("No document vectors found in ChromaDB.")
        return

    stored_data = collection.get(
        limit=total_chunks,
        include=["embeddings"]
    )

    embeddings = np.asarray(stored_data["embeddings"])
    ids = stored_data["ids"]

    print(f"Total stored vectors: {len(embeddings)}")

    # -----------------------------------------
    # 2. Prepare query vector
    # -----------------------------------------

    query_vector = np.asarray(query_embedding).reshape(1, -1)

    # -----------------------------------------
    # 3. Get retrieved top-k vectors
    # -----------------------------------------

    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]

    # -----------------------------------------
    # 4. Combine all document vectors + query
    # -----------------------------------------

    all_vectors = np.vstack([
        embeddings,
        query_vector
    ])

    if all_vectors.shape[0] < 2 or all_vectors.shape[1] < 2:
        print("Not enough data for 2D PCA visualization.")
        return

    # -----------------------------------------
    # 5. Reduce dimensions using PCA
    # -----------------------------------------

    pca = PCA(n_components=2)

    vectors_2d = pca.fit_transform(all_vectors)

    document_points = vectors_2d[:-1]
    query_point = vectors_2d[-1]

    # Map document IDs to their 2D coordinates
    id_to_position = {
        doc_id: document_points[i]
        for i, doc_id in enumerate(ids)
    }

    # -----------------------------------------
    # 6. Create visualization
    # -----------------------------------------

    fig, ax = plt.subplots(figsize=(14, 9))

    # Show ALL document vectors
    ax.scatter(
        document_points[:, 0],
        document_points[:, 1],
        color="lightgray",
        alpha=0.7,
        s=55,
        label=f"All document vectors ({len(embeddings)})"
    )

    # -----------------------------------------
    # 7. Highlight retrieved vectors
    # -----------------------------------------

    for rank, (doc_id, distance) in enumerate(
        zip(retrieved_ids, distances),
        start=1
    ):

        if doc_id not in id_to_position:
            continue

        point = id_to_position[doc_id]

        ax.scatter(
            point[0],
            point[1],
            s=160,
            marker="o",
            edgecolors="black",
            linewidths=1,
            label=f"Retrieved vector {rank}"
        )

        # Connect query to retrieved vector
        ax.plot(
            [query_point[0], point[0]],
            [query_point[1], point[1]],
            linestyle="--",
            alpha=0.6
        )

        # Show original ChromaDB distance
        ax.annotate(
            f"#{rank} | D={distance:.3f}",
            (point[0], point[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9
        )

    # -----------------------------------------
    # 8. Show user query vector
    # -----------------------------------------

    ax.scatter(
        query_point[0],
        query_point[1],
        color="red",
        marker="*",
        s=400,
        edgecolors="black",
        label="User query",
        zorder=5
    )

    ax.annotate(
        "USER QUERY",
        query_point,
        xytext=(10, 12),
        textcoords="offset points",
        fontweight="bold"
    )

    # -----------------------------------------
    # 9. Graph information
    # -----------------------------------------

    variance = pca.explained_variance_ratio_.sum() * 100

    ax.set_title(
        "Naive RAG - Complete Vector Space Visualization\n"
        f"Total vectors: {len(embeddings)} | "
        f"PCA explained variance: {variance:.2f}%",
        fontsize=14
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
