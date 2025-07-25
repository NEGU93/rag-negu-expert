import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


def tsne_visualizer(collection):
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    metadatas = result["metadatas"]
    doc_types = set([metadata["doc_type"] for metadata in metadatas])
    print(
        f"Visualizing {len(vectors)} vectors with {len(doc_types)} document types"
    )
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                mode="markers",
                marker=dict(size=5, opacity=0.8),
                text=[
                    f"Type: {t}<br>Text: {d[:100]}..."
                    for t, d in zip(doc_types, documents)
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="2D Chroma Vector Store Visualization",
        scene=dict(xaxis_title="x", yaxis_title="y"),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40),
    )

    fig.show()
