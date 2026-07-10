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
