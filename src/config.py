from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExperimentConfig:
    # Dataset
    num_classes: int = 10

    # Active learning
    initial_labelled_size: int = 0
    query_budget: int = 10
    num_al_rounds: int = 5

    # TypiClust
    num_neighbours: int = 20
    min_cluster_size: int = 5
    max_clusters: int = 500

    # Representation learning
    embedding_dim: int = 512
    projection_dim: int = 128

    # SimCLR pretraining
    simclr_epochs: int = 50
    simclr_batch_size: int = 512
    simclr_learning_rate: float = 0.4
    simclr_momentum: float = 0.9
    simclr_weight_decay: float = 1e-4
    temperature: float = 0.5

    # SCAN clustering
    scan_epochs: int = 50
    scan_batch_size: int = 512
    scan_learning_rate: float = 0.1
    scan_entropy_weight: float = 2.0

    # Reproducibility
    seed: int = 42

    @property
    def cumulative_budgets(self):
        return [
            self.initial_labelled_size
            + self.query_budget * round_number
            for round_number in range(1, self.num_al_rounds + 1)
        ]

    def to_dict(self):
        return asdict(self)
