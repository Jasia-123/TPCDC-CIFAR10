from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for TPCDC on CIFAR-10.

    TPCDC uses the SCAN pipeline for deep clustering. It must not
    use K-means for the final cluster assignments.
    """

    # Supervised evaluation

    classifier_epochs: int = 100
    classifier_batch_size: int = 128
    classifier_learning_rate: float = 0.025
    classifier_momentum: float = 0.9
    classifier_weight_decay: float = 5e-4

    # Dataset

    num_classes: int = 10
    num_training_samples: int = 50_000
    embedding_dim: int = 512

    # Active learning protocol

    initial_labelled_size: int = 0
    query_budget: int = 10
    num_al_rounds: int = 5

    # TypiClust selection

    typicality_neighbours: int = 20
    min_cluster_size: int = 5
    max_clusters: int = 500

    # SCAN representation pretraining

    pretrain_epochs: int = 50
    pretrain_batch_size: int = 512
    pretrain_learning_rate: float = 0.4
    pretrain_momentum: float = 0.9
    pretrain_weight_decay: float = 1e-4
    pretrain_temperature: float = 0.5
    projection_dim: int = 128

    # Nearest-neighbour mining for SCAN

    scan_neighbours: int = 20

    # SCAN clustering stage
    #
    # The original SCAN paper trains this stage for 100 epochs.
    # We use 50 epochs because of Colab resource limits.
    # This deviation will be discussed in the report.

    scan_epochs: int = 50
    scan_batch_size: int = 128
    scan_optimizer: str = "adam"
    scan_learning_rate: float = 1e-4
    scan_momentum: float = 0.9
    scan_weight_decay: float = 1e-4
    scan_entropy_weight: float = 5.0

    # Data loading

    num_workers: int = 2

    # Reproducibility

    seed: int = 42

    @property
    def cumulative_budgets(self) -> list[int]:
        """Return the labelled-set size after each AL round."""
        return [
            self.initial_labelled_size
            + self.query_budget * round_number
            for round_number in range(1, self.num_al_rounds + 1)
        ]

    def cluster_count(self, labelled_size: int) -> int:
        """
        Return the number of SCAN output clusters for one AL round.

        Paper rule:
            K = min(|L| + B, max_clusters)
        """
        if labelled_size < 0:
            raise ValueError("labelled_size must be non-negative.")

        return min(
            labelled_size + self.query_budget,
            self.max_clusters,
        )

    def validate(self) -> None:
        """Validate important configuration constraints."""
        if self.query_budget <= 0:
            raise ValueError("query_budget must be positive.")

        if self.num_al_rounds <= 0:
            raise ValueError("num_al_rounds must be positive.")

        if self.typicality_neighbours <= 0:
            raise ValueError("typicality_neighbours must be positive.")

        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1.")

        if self.max_clusters < self.query_budget:
            raise ValueError(
                "max_clusters must be at least as large as query_budget."
            )

        if self.pretrain_epochs <= 0 or self.scan_epochs <= 0:
            raise ValueError("Training epochs must be positive.")

        if self.classifier_epochs <= 0:
            raise ValueError(
                "classifier_epochs must be positive."
            )

        if self.classifier_batch_size <= 0:
            raise ValueError(
                "classifier_batch_size must be positive."
            )

    def to_dict(self) -> dict:
        return asdict(self)
