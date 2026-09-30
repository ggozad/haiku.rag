from collections.abc import Mapping

import numpy as np
from sklearn.manifold import TSNE

MIN_DOCUMENTS = 3


def project(centroids: Mapping[str, bytes]) -> dict[str, tuple[float, float]]:
    """t-SNE map positions of documents, by id; documents with one centroid share a point."""
    if len(centroids) < MIN_DOCUMENTS:
        return {}
    ids = list(centroids)
    unit = np.stack([np.frombuffer(centroids[i], dtype="<f4") for i in ids])
    distinct, owner = np.unique(unit, axis=0, return_inverse=True)
    if len(distinct) < MIN_DOCUMENTS:
        points = np.array([(float(i), 0.0) for i in range(len(distinct))])
    else:
        points = TSNE(
            n_components=2,
            metric="cosine",
            # Neighbourhoods span 3 x perplexity points, which must leave some out.
            perplexity=min(30, (len(distinct) - 1) / 3),
            init="pca",
            random_state=0,
        ).fit_transform(distinct)
    return {
        doc_id: (float(points[i][0]), float(points[i][1]))
        for doc_id, i in zip(ids, owner.reshape(-1))
    }
