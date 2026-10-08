"""Run from the repo root: PYTHONPATH=src python examples/windowed_analytics.py.

Synthetic numeric data only. Domain-specific state/action construction, source
identity, physical time mapping, and plotting remain application concerns.
"""

import numpy as np

from pyidk import (
    IsolationKernel,
    PointFeatureCache,
    SequenceBatch,
    Standardizer,
    TransitionRepresentation,
    reduce_references,
)


def features(states: SequenceBatch, dimension: int) -> SequenceBatch:
    """Example external feature construction for 4D, 8D, and 9D sequences."""
    if dimension == 4:
        return states
    transitions = TransitionRepresentation().transform(states)
    if dimension == 8:
        return transitions
    # An illustrative scalar input per transition, not a CartPole action encoder.
    return SequenceBatch.from_sequences(
        np.column_stack((transitions.sequence(i), np.tanh(states.sequence(i)[:-1, 0])))
        for i in range(states.n_sequences)
    )


def main() -> None:
    rng = np.random.default_rng(42)

    def population(lengths: list[int]) -> SequenceBatch:
        return SequenceBatch.from_sequences(
            np.cumsum(rng.normal(scale=0.1, size=(n, 4)), axis=0) for n in lengths
        )

    fitting_states = population([70, 80, 65])
    query_states = population([55, 42])
    query_states.values[25:40] += 2.0
    independent_references = population([58, 47])
    reference_states = SequenceBatch.from_sequences(
        [query_states.sequence(0).copy()] + [independent_references.sequence(i) for i in range(2)]
    )
    # The first reference is copied from query source 100 and must be excluded
    # for that query even though it resides in a separate batch/sequence.
    query_source_ids = np.array([100, 200])
    reference_source_ids = np.array([100, 300, 400])

    for dimension in (4, 8, 9):
        fit = features(fitting_states, dimension)
        scaler = Standardizer().fit(fit.values)
        kernel = IsolationKernel(n_partitions=24, samples_per_partition=8, random_state=42).fit(
            scaler.transform(fit.values)
        )

        def cache(
            states: SequenceBatch,
            dimension: int = dimension,
            scaler: Standardizer = scaler,
            kernel: IsolationKernel = kernel,
        ) -> PointFeatureCache:
            represented = features(states, dimension)
            scaled = represented.with_values(scaler.transform(represented.values))
            return PointFeatureCache.from_sequences(kernel, scaled)

        query_cache, reference_cache = cache(query_states), cache(reference_states)
        for transitions_per_window in (4, 8, 12):
            # For k physical transitions: k+1 state rows, or k transition rows.
            length = transitions_per_window + (dimension == 4)
            references = reference_cache.windows(length, stride=3)
            reference_sources = reference_source_ids[references.sequence_index]
            tidy_rows = []
            for query in query_cache.iter_windows(length, stride=2, batch_size=16):
                query_sources = query_source_ids[query.sequence_index]

                def eligible(
                    q, r, query_sources=query_sources, reference_sources=reference_sources
                ):
                    return query_sources[q, None] != reference_sources[None, r]

                for metric in ("similarity", "distance"):
                    for aggregation in ("mean", "top_k"):
                        result = reduce_references(
                            query,
                            references,
                            metric=metric,
                            aggregation=aggregation,
                            k=5 if aggregation == "top_k" else None,
                            eligibility=eligible,
                            query_block_size=8,
                            reference_block_size=16,
                            return_selected=aggregation == "top_k",
                        )
                        # Selected reference indices are relative to this reference bank.
                        if result.selected_indices is not None and len(query.intervals):
                            matched_intervals = references.intervals[result.selected_indices[0]]
                            assert len(matched_intervals) == result.selected_count[0]
                        for row in result.records():
                            # Last raw state consumed: transition row i consumes state i+1.
                            row["endpoint"] = row["stop"] - (dimension == 4)
                            tidy_rows.append(row)
                support = query.outside_mass()
                for i, (sequence, start, stop) in enumerate(query.intervals):
                    tidy_rows.append(
                        dict(
                            query_sequence=int(sequence),
                            start=int(start),
                            stop=int(stop),
                            row_count=int(stop - start),
                            endpoint=int(stop) - (dimension == 4),
                            metric="outside_mass",
                            aggregation="partition_mean",
                            score=float(support.mean[i]),
                        )
                    )

            # IQR across trajectories at a common endpoint, after reference reduction.
            # Outside-mass IQR also uses one mean-support score per trajectory.
            for metric, aggregation in [
                ("similarity", "top_k"),
                ("outside_mass", "partition_mean"),
            ]:
                selected = [
                    r
                    for r in tidy_rows
                    if r["metric"] == metric and r["aggregation"] == aggregation
                ]
                summaries = []
                for endpoint in sorted({r["endpoint"] for r in selected}):
                    rows = [
                        r for r in selected if r["endpoint"] == endpoint and np.isfinite(r["score"])
                    ]
                    scores = np.array([r["score"] for r in rows])
                    if len(scores):
                        summaries.append(
                            (endpoint, len(rows), np.quantile(scores, [0.25, 0.5, 0.75]))
                        )
                # Show both a shared endpoint and the endpoint with fewer contributors.
                for endpoint, count, quantiles in (summaries[0], summaries[-1]):
                    print(
                        f"d={dimension}, k={transitions_per_window}, {metric}: "
                        f"endpoint={endpoint}, contributors={count}, "
                        f"quartiles={np.round(quantiles, 4).tolist()}"
                    )

            # Query and reference durations may differ without rebuilding either cache.
            alternate = reference_cache.aggregate([(0, 1, length + 3), (1, 4, length + 6)])
            assert alternate.row_count.tolist() == [length + 2, length + 2]


if __name__ == "__main__":
    main()
