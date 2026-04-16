from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, Callable

import pandas as pd
import numpy as np

ConstraintChecker = Callable[[pd.DataFrame], bool]

class MondrianAnonymizer:
    def __init__(
        self,
        data: pd.DataFrame,
        k: float = 10.0,
        l: float = 2.0,
        t: float = 0.2,
        qi_continuous=None,
        qi_categorical=None,
        sensitive_col=None,
        constraint_check_callbacks: Iterable[Callable] | None = None,
    ):
        """
        Initializes the Mondrian Anonymizer.

        :param data: Pandas DataFrame to anonymize.
        :param k: Minimum number of records per bucket (k-anonymity).
        :param l: Minimum number of distinct sensitive values per bucket (l-diversity).
        :param t: Maximum allowed distance from the global sensitive distribution (t-closeness).
        :param qi_continuous: List of continuous quasi-identifier column names.
        :param qi_categorical: List of categorical quasi-identifier column names.
        :param sensitive_col: The name of the sensitive attribute column.
        :param constraint_check_callbacks:
            Optional iterable of custom constraint check functions.
            These determine when the splits stop.
        """
        self.df = data
        self.k = max(k, 0.0)
        self.l = max(l, 0.0)
        self.t = max(t, 0.0)
        self.qi_continuous = qi_continuous or ()
        self.qi_categorical = qi_categorical or ()
        self.sensitive_col = sensitive_col
        self.constraint_check_callbacks: Iterable[ConstraintChecker] = constraint_check_callbacks or [
            self.check_k_anonimity,
            self.check_l_divergence_ENTROPY,
            self.check_t_closeness_TVD,
        ]
        # Calculate global distribution for t-closeness
        self.global_freqs: dict[Any, float] = self.df[self.sensitive_col].value_counts(normalize=True).to_dict()

    def _get_spans(self, data: pd.DataFrame) -> dict[Any, float | int]:
        """Calculates the span (max-min or unique count) for each QI."""
        spans = {}
        for col in self.qi_continuous:
            spans[col] = data[col].max() - data[col].min()
        for col in self.qi_categorical:
            spans[col] = data[col].nunique()
        return spans

    @staticmethod
    def split_data(data, column, *, is_categorical: bool = False):
        """Splits the dataframe into two partitions based on the median/set division."""
        if is_categorical:
            unique_vals = list(data[column].unique())
            # Basic categorical heuristic: split the unique values in half
            mid = len(unique_vals) // 2
            lhs_vals = unique_vals[:mid]

            lhs = data[data[column].isin(lhs_vals)]
            rhs = data[~data[column].isin(lhs_vals)]
        else:
            # Sort and split by index to perfectly divide records and handle identical medians
            sorted_data = data.sort_values(by=column)
            mid = len(sorted_data) // 2
            lhs = sorted_data.iloc[:mid]
            rhs = sorted_data.iloc[mid:]

        return lhs, rhs

    def check_k_anonimity(self, partition):
        """Checks if the partition satisfies k-anonymity."""
        return len(partition) >= self.k

    def k_anonimity(self, partition):
        return len(partition)

    def check_l_divergence_ENTROPY(self, partition):
        """Checks if the partition satisfies l-diversity."""
        if not self.l > 0:
            return True

        partition_size = len(partition)
        sensitive_feat_frequencies: np.ndarray = partition[self.sensitive_col].value_counts().values
        entropy_N = partition_size * np.log2(partition_size) - np.sum(sensitive_feat_frequencies * np.log2(sensitive_feat_frequencies))
        return entropy_N >= partition_size * np.log2(self.l)

    def check_t_closeness_TVD(self, partition):
        """Checks if the partition satisfies t-closeness with Total Variation Distance"""
        if not self.t < 1.0:
            return True
        local_sensitive_distribution = partition[self.sensitive_col].value_counts(normalize=True).to_dict()
        tvd = 0.5 * sum(
            abs(local_sensitive_distribution.get(val, 0) - self.global_freqs.get(val, 0))
            for val in self.global_freqs.keys()
        )
        return tvd <= self.t

    def _is_valid(self, partition: pd.DataFrame) -> bool:
        """Checks if a partition satisfies k, l, and t constraints."""
        return all(
            map(lambda check: check(partition=partition), self.constraint_check_callbacks)
        )

    def variance_heuristic(self, data: pd.DataFrame) -> list[tuple[Any, float | int]]:
        """Returns a list of tuples (column, variance) sorted by descending variance."""
        spans = self._get_spans(data)
        # Sort dimensions by span descending to pick the dimension with the highest variance
        return sorted(spans.items(), key=lambda x: x[1], reverse=True)

    def _anonymize_recursive(self, data: pd.DataFrame) -> list[pd.DataFrame]:
        """Recursively partitions the dataset using the greedy Mondrian heuristic."""
        heuristic_dims_order = self.variance_heuristic(data)

        for dim, span in heuristic_dims_order:
            is_cat = dim in self.qi_categorical

            # Skip dimensions that cannot be split further
            if (is_cat and span <= 1) or (not is_cat and np.allclose(span, 0.0)):
                continue

            lhs, rhs = self.split_data(data, dim, is_categorical=is_cat)

            # Check if BOTH resulting partitions satisfy constraints
            if self._is_valid(lhs) and self._is_valid(rhs):
                return self._anonymize_recursive(lhs) + self._anonymize_recursive(rhs)

        # no valid sub-partitions, return the current data as a leaf bucket
        return [data]

    def _generalize_partitions(self, partitions):
        """Replaces QI values in each bucket with generalized cartesian products."""
        anonymized_rows = []
        for p in partitions:
            summary = {}
            # Generalize Continuous
            for col in self.qi_continuous:
                min_val, max_val = p[col].min(), p[col].max()
                summary[col] = f"[{min_val} - {max_val}]" if min_val != max_val else str(min_val)

            # Generalize Categorical
            for col in self.qi_categorical:
                summary[col] = " | ".join(map(str, p[col].unique()))

            # Apply generalized summaries to the bucket rows
            for _, row in p.iterrows():
                new_row = row.copy()
                for col in summary:
                    new_row[col] = summary[col]
                anonymized_rows.append(new_row)

        return pd.DataFrame(anonymized_rows)

    def run(self):
        """Executes the anonymization and returns the generalized dataframe."""
        # Check if the initial dataset satisfies the baseline constraints
        if not self._is_valid(self.df):
            raise ValueError(
                "The initial dataset does not satisfy the baseline k, l, or t constraints. Relax parameters.")

        final_partitions = self._anonymize_recursive(self.df)
        return self._generalize_partitions(final_partitions), final_partitions