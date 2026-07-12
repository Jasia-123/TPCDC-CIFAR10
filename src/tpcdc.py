from dataclasses import asdict, dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class QuerySelectionRecord:
    """
    Diagnostic information for one selected TPCDC query.
    """

    selection_position: int
    selected_index: int
    cluster_id: int
    cluster_size: int
    labelled_count_before_selection: int
    typicality: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_tpcdc_inputs(
    cluster_assignments: np.ndarray,
    features: np.ndarray,
    labelled_indices: list[int] | np.ndarray,
    query_budget: int,
) -> None:
    """
    Validate arrays and indices before TPCDC selection.
    """

    if cluster_assignments.ndim != 1:
        raise ValueError(
            "cluster_assignments must be one-dimensional."
        )

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [samples, dimensions]."
        )

    if len(cluster_assignments) != features.shape[0]:
        raise ValueError(
            "Assignments and features must contain "
            "the same number of samples."
        )

    if query_budget <= 0:
        raise ValueError(
            "query_budget must be positive."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "features contain NaN or infinite values."
        )

    labelled_array = np.asarray(
        labelled_indices,
        dtype=np.int64,
    )

    if labelled_array.size > 0:
        if labelled_array.min() < 0:
            raise ValueError(
                "labelled indices cannot be negative."
            )

        if labelled_array.max() >= features.shape[0]:
            raise ValueError(
                "A labelled index exceeds the dataset size."
            )

        if np.unique(labelled_array).size != labelled_array.size:
            raise ValueError(
                "labelled_indices contains duplicates."
            )


def build_cluster_members(
    cluster_assignments: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Map every occupied cluster ID to its sample indices.
    """

    cluster_members: dict[int, np.ndarray] = {}

    for cluster_id in np.unique(cluster_assignments):
        members = np.flatnonzero(
            cluster_assignments == cluster_id
        ).astype(np.int64)

        cluster_members[int(cluster_id)] = members

    return cluster_members


def compute_cluster_typicalities(
    features: np.ndarray,
    cluster_indices: np.ndarray,
    max_neighbours: int = 20,
    epsilon: float = 1e-12,
) -> dict[int, float]:
    """
    Compute typicality for every point in one cluster.

    Typicality is the inverse mean Euclidean distance to the
    nearest neighbours inside the same cluster.

    The point itself is excluded, so the number of neighbours is:
        min(max_neighbours, cluster_size - 1)
    """

    if max_neighbours <= 0:
        raise ValueError(
            "max_neighbours must be positive."
        )

    cluster_indices = np.asarray(
        cluster_indices,
        dtype=np.int64,
    )

    cluster_size = cluster_indices.size

    if cluster_size < 2:
        raise ValueError(
            "At least two samples are required "
            "to compute typicality."
        )

    cluster_features = features[cluster_indices]

    neighbour_count = min(
        max_neighbours,
        cluster_size - 1,
    )

    nearest_neighbours = NearestNeighbors(
        n_neighbors=neighbour_count + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    nearest_neighbours.fit(
        cluster_features
    )

    distances, indices = (
        nearest_neighbours.kneighbors(
            cluster_features,
            return_distance=True,
        )
    )

    # The first neighbour should be the point itself.
    neighbour_distances = distances[:, 1:]

    if neighbour_distances.shape != (
        cluster_size,
        neighbour_count,
    ):
        raise RuntimeError(
            "Unexpected nearest-neighbour "
            "distance shape."
        )

    mean_distances = neighbour_distances.mean(
        axis=1,
    )

    typicalities = 1.0 / (
        mean_distances + epsilon
    )

    return {
        int(dataset_index): float(typicality)
        for dataset_index, typicality in zip(
            cluster_indices,
            typicalities,
        )
    }


def select_tpcdc_queries(
    cluster_assignments: np.ndarray,
    features: np.ndarray,
    labelled_indices: list[int] | np.ndarray,
    query_budget: int,
    min_cluster_size: int = 5,
    max_typicality_neighbours: int = 20,
) -> tuple[list[int], list[QuerySelectionRecord]]:
    """
    Select one TPCDC query batch.

    At each selection step:

      1. Exclude clusters with fewer than min_cluster_size samples.
      2. Find clusters containing the fewest currently labelled points.
      3. Among those clusters, choose the largest cluster that still
         contains an unlabelled sample.
      4. Select the most typical unlabelled sample in that cluster.
      5. Treat the selected sample as labelled before choosing the
         next sample.

    Labels themselves are never accessed.
    """

    validate_tpcdc_inputs(
        cluster_assignments=cluster_assignments,
        features=features,
        labelled_indices=labelled_indices,
        query_budget=query_budget,
    )

    if min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    cluster_members = build_cluster_members(
        cluster_assignments
    )

    working_labelled = set(
        int(index)
        for index in labelled_indices
    )

    selected_queries: list[int] = []
    selection_records: list[
        QuerySelectionRecord
    ] = []

    # Cache typicality because assignments and features stay
    # unchanged throughout this query round.
    typicality_cache: dict[
        int,
        dict[int, float],
    ] = {}

    for selection_position in range(
        query_budget
    ):
        eligible_clusters = []

        for cluster_id, members in cluster_members.items():
            cluster_size = len(members)

            # Paper wording is slightly ambiguous:
            # - clusters with fewer than 5 samples are dropped;
            # - another sentence says clusters should be larger than 5.
            #
            # We follow the explicit "drop fewer than 5" rule, so clusters
            # of exactly 5 samples remain eligible.
            if cluster_size < min_cluster_size:
                continue

            unlabelled_members = [
                int(index)
                for index in members
                if int(index) not in working_labelled
            ]

            if not unlabelled_members:
                continue

            labelled_count = sum(
                int(index) in working_labelled
                for index in members
            )

            eligible_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "members": members,
                    "cluster_size": cluster_size,
                    "labelled_count": labelled_count,
                    "unlabelled_members": unlabelled_members,
                }
            )

        if not eligible_clusters:
            raise RuntimeError(
                "TPCDC could not find an eligible cluster. "
                "Inspect SCAN cluster sizes or reduce "
                "min_cluster_size."
            )

        minimum_labelled_count = min(
            cluster["labelled_count"]
            for cluster in eligible_clusters
        )

        least_covered_clusters = [
            cluster
            for cluster in eligible_clusters
            if cluster["labelled_count"]
            == minimum_labelled_count
        ]

        # Largest cluster first. Cluster ID is used as a
        # deterministic tie-breaker.
        selected_cluster = sorted(
            least_covered_clusters,
            key=lambda cluster: (
                -cluster["cluster_size"],
                cluster["cluster_id"],
            ),
        )[0]

        cluster_id = selected_cluster[
            "cluster_id"
        ]

        if cluster_id not in typicality_cache:
            typicality_cache[cluster_id] = (
                compute_cluster_typicalities(
                    features=features,
                    cluster_indices=selected_cluster[
                        "members"
                    ],
                    max_neighbours=(
                        max_typicality_neighbours
                    ),
                )
            )

        cluster_typicalities = (
            typicality_cache[cluster_id]
        )

        unlabelled_members = selected_cluster[
            "unlabelled_members"
        ]

        # Highest typicality first. Dataset index is a
        # deterministic tie-breaker.
        selected_index = sorted(
            unlabelled_members,
            key=lambda index: (
                -cluster_typicalities[index],
                index,
            ),
        )[0]

        selected_typicality = (
            cluster_typicalities[selected_index]
        )

        selected_queries.append(
            selected_index
        )

        selection_records.append(
            QuerySelectionRecord(
                selection_position=(
                    selection_position + 1
                ),
                selected_index=selected_index,
                cluster_id=cluster_id,
                cluster_size=selected_cluster[
                    "cluster_size"
                ],
                labelled_count_before_selection=(
                    selected_cluster[
                        "labelled_count"
                    ]
                ),
                typicality=selected_typicality,
            )
        )

        # This affects coverage for the remaining selections
        # in the same active-learning round.
        working_labelled.add(
            selected_index
        )

    if len(selected_queries) != query_budget:
        raise RuntimeError(
            "TPCDC returned the wrong number of queries."
        )

    if len(set(selected_queries)) != query_budget:
        raise RuntimeError(
            "TPCDC selected duplicate queries."
        )

    if set(selected_queries).intersection(
        set(int(index) for index in labelled_indices)
    ):
        raise RuntimeError(
            "TPCDC selected an already-labelled sample."
        )

    return selected_queries, selection_records


def summarize_clusters(
    cluster_assignments: np.ndarray,
    expected_num_clusters: int,
) -> dict:
    """
    Produce useful SCAN cluster diagnostics.
    """

    unique_clusters, counts = np.unique(
        cluster_assignments,
        return_counts=True,
    )

    return {
        "expected_clusters": int(
            expected_num_clusters
        ),
        "occupied_clusters": int(
            unique_clusters.size
        ),
        "empty_clusters": int(
            expected_num_clusters
            - unique_clusters.size
        ),
        "smallest_occupied_cluster": int(
            counts.min()
        ),
        "largest_occupied_cluster": int(
            counts.max()
        ),
        "mean_occupied_cluster_size": float(
            counts.mean()
        ),
        "median_occupied_cluster_size": float(
            np.median(counts)
        ),
    }
