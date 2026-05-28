import unittest
from ucimlrepo import fetch_ucirepo

from anonimize_data import MondrianAnonymizer

import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter


class TestTClosenessComparison(unittest.TestCase):
    
    @staticmethod
    def get_group_distribution(partitions):
        """
        Calculate group sizes and their frequencies.
        
        :param partitions: List of DataFrame partitions
        :return: Tuple of (sizes, frequencies)
        """
        group_sizes = [len(group) for group in partitions]
        size_counts = Counter(group_sizes)
        sizes = sorted(size_counts.keys())
        frequencies = [size_counts[s] for s in sizes]
        return sizes, frequencies
    
    def test_t_closeness_enabled_vs_disabled(self):
        """
        Compare MondrianAnonymizer behavior with t-closeness enabled vs disabled.
        """
        # 1. Import dataset
        print("Starting download of UCI dataset (id=2)...", flush=True)
        start_time = time.time()
        adult = fetch_ucirepo(id=2)
        elapsed = time.time() - start_time
        print(f"Finished download in {elapsed:.1f}s", flush=True)

        X = adult.data.features
        y = adult.data.targets

        df = pd.concat([X, y], axis=1).dropna()
        df.columns = df.columns.str.strip()
        df['income'] = df['income'].str.strip()
        
        params = {
            'k': 5,
            'l': 2,
            'tau': 0.2,
            't': 0.2,
            'qi_continuous': ['age'],
            'qi_categorical': ['sex', 'race'],
            'sensitive_col': 'occupation'
        }
        
        print(f"\n{'='*60}")
        print(f"Test: t-closeness enabled vs disabled")
        print(f"Parameters: {params}")
        print(f"{'='*60}\n")
        
        # Instance 1: WITH t-closeness checks
        print("Instance 1: WITH t-closeness checks (enable_t_closeness=True)")
        print("-" * 60)
        anonymizer_with_t = MondrianAnonymizer(
            data=df,
            **params,
            enable_t_closeness=True,
            plot_filename='t_closeness_enabled.png'
        )
        
        anonymized_df_with_t, partitions_with_t = anonymizer_with_t.run()
        
        print(f"Number of partitions: {len(partitions_with_t)}")
        sizes_with_t, freq_with_t = self.get_group_distribution(partitions_with_t)
        print(f"Group sizes distribution: {dict(zip(sizes_with_t, freq_with_t))}")
        
        # Verify constraints
        assert all(anonymizer_with_t.check_k_anonimity(partition) for partition in partitions_with_t), \
            "k-anonymity violated"
        assert all(anonymizer_with_t.check_l_divergence_ENTROPY(partition) for partition in partitions_with_t), \
            "l-diversity violated"
        assert all(anonymizer_with_t.check_t_closeness_TVD(partition) for partition in partitions_with_t), \
            "t-closeness violated"
        print("✓ All constraints satisfied (k-anonymity, l-diversity, t-closeness)\n")
        
        # Instance 2: WITHOUT t-closeness checks
        print("Instance 2: WITHOUT t-closeness checks (enable_t_closeness=False)")
        print("-" * 60)
        anonymizer_without_t = MondrianAnonymizer(
            data=df,
            **params,
            enable_t_closeness=False,
            plot_filename='t_closeness_disabled.png'
        )
        
        anonymized_df_without_t, partitions_without_t = anonymizer_without_t.run()
        
        print(f"Number of partitions: {len(partitions_without_t)}")
        sizes_without_t, freq_without_t = self.get_group_distribution(partitions_without_t)
        print(f"Group sizes distribution: {dict(zip(sizes_without_t, freq_without_t))}")
        
        # Verify constraints (should not check t-closeness)
        assert all(anonymizer_without_t.check_k_anonimity(partition) for partition in partitions_without_t), \
            "k-anonymity violated"
        assert all(anonymizer_without_t.check_l_divergence_ENTROPY(partition) for partition in partitions_without_t), \
            "l-diversity violated"
        print("✓ All constraints satisfied (k-anonymity, l-diversity)")
        print("✗ t-closeness NOT checked\n")
        
        # Comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"With t-closeness:    {len(partitions_with_t)} partitions")
        print(f"Without t-closeness: {len(partitions_without_t)} partitions")
        print(f"Difference: {len(partitions_without_t) - len(partitions_with_t)} partitions")
        print(f"{'='*60}\n")
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 6), dpi=150)
        
        # Plot 1: With t-closeness
        ax1.bar(range(len(sizes_with_t)), freq_with_t, color='steelblue')
        ax1.set_xlabel('Group Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'With t-closeness (n={len(partitions_with_t)} partitions)')
        ax1.set_xscale('log')
        ax1.set_xticks(range(len(sizes_with_t)))
        ax1.set_xticklabels(sizes_with_t)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Without t-closeness
        ax2.bar(range(len(sizes_without_t)), freq_without_t, color='coral')
        ax2.set_xlabel('Group Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Without t-closeness (n={len(partitions_without_t)} partitions)')
        ax2.set_xscale('log')
        ax2.set_xticks(range(len(sizes_without_t)))
        ax2.set_xticklabels(sizes_without_t)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('t_closeness_comparison.png', dpi=150)
        print("✓ Saved comparison plot: t_closeness_comparison.png")


if __name__ == '__main__':
    unittest.main()
