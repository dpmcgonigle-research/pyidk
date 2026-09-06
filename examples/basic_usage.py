"""Compare three short episodes using transition-based IDK embeddings.

Each episode contains four two-dimensional observations. The first two follow
similar paths, while the third visits a different region and moves in a different
direction. SequenceBatch preserves episode boundaries so that constructing
transitions never joins observations from separate episodes.

TransitionRepresentation concatenates consecutive observations into four-feature
vectors, producing three transitions per episode. Standardizer then scales each
feature using the pooled transitions. IsolationKernel fits 100 random partitions
with four sampled centers each, giving a sparse 400-feature point embedding.
IsolationDistributionalKernel averages these features within each episode to
represent its distribution of transitions. Transitions retain local direction,
but averaging them does not retain their global temporal order.

The printed 3-by-3 matrix contains pairwise episode similarities. The first two
episodes should resemble each other more than the third. These are raw kernel
similarities, not probabilities; diagonal entries need not equal one. The fixed
seed makes the example reproducible. For this illustration, scaling and the
kernel are fitted on all three episodes; an evaluation would fit them on training
episodes only and transform held-out episodes with those fitted objects.
"""

import numpy as np

from pyidk import (
    IsolationDistributionalKernel,
    IsolationKernel,
    SequenceBatch,
    Standardizer,
    TransitionRepresentation,
)

episodes = SequenceBatch.from_sequences(
    [
        np.array([[0.0, 0.0], [0.1, 1.0], [0.2, 1.5], [0.3, 2.0]]),
        np.array([[0.0, 0.1], [0.1, 0.9], [0.2, 1.6], [0.3, 2.1]]),
        np.array([[2.0, 5.0], [2.1, 4.5], [2.2, 4.0], [2.3, 3.5]]),
    ]
)

representation = TransitionRepresentation(lags=(0, 1))
transitions = representation.transform(episodes)

scaler = Standardizer().fit(transitions.values)
transitions = transitions.with_values(scaler.transform(transitions.values))

idk = IsolationDistributionalKernel(
    IsolationKernel(
        n_partitions=100,
        samples_per_partition=4,
        random_state=42,
    )
)

embeddings = idk.fit_transform(transitions)
similarity = idk.similarity(embeddings)

print(similarity)
