import unittest
from ucimlrepo import fetch_ucirepo

from anonimize_data import MondrianAnonymizer

import pandas as pd

# wykresy
import matplotlib.pyplot as plt
from collections import Counter

class MyTestCase(unittest.TestCase):
    
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
    
    @staticmethod
    def plot_distributions(distributions_dict):
        """
        Plot multiple lines on one graph for different parameter combinations.
        
        :param distributions_dict: Dict with keys like "l=2, t=0.2" and values (sizes, frequencies)
        """
        plt.figure(figsize=(20, 6))
        
        for label, (sizes, frequencies) in distributions_dict.items():
            plt.plot(sizes, frequencies, marker='o', linestyle='-', label=label)
        
        plt.xlabel("Rozmiar grupy EC")
        plt.ylabel("Liczba grup")
        plt.title("Dystrybucja liczności grup — porównanie dla różnych l i t")
        plt.xscale('log')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("dystrybucja_licznosci_grup_comparison.png")
        print("✓ Saved comparison plot: dystrybucja_licznosci_grup_comparison.png")
    
    def test_mondrian_anonimizer(self):
        
        # 1. Import dataset
        adult = fetch_ucirepo(id=2)

        X = adult.data.features
        y = adult.data.targets

        df = pd.concat([X, y], axis=1).dropna()
        df.columns = df.columns.str.strip()
        df['income'] = df['income'].str.strip()
        
        # Test different combinations of l and t parameters
        l_values = [2]        # l-diversity levels
        t_values = [0.2, 0.4, 0.6, 0.8]  # t-closeness thresholds
        
        distributions = {}
        dist_lengths = {}

        for l in l_values:
            for t in t_values:
                print(f"\n{'='*60}")
                print(f"Testing: l={l}, t={t}")
                print(f"{'='*60}")
                
                # Configure the constraints with current l and t
                anonymizer = MondrianAnonymizer(
                    data=df,
                    k=5,
                    l=l,
                    tau=0.2,
                    t=t,
                    qi_continuous=['age'],
                    qi_categorical=['sex', 'race'],
                    sensitive_col='occupation'
                )

                # Run Anonymization
                anonymized_df, partitions = anonymizer.run()

                print(f"Number of partitions: {len(partitions)}")
                
                # Collect distribution data
                sizes, frequencies = self.get_group_distribution(partitions)
                distributions[f"l={l}, t={t}"] = (sizes, frequencies)
                dist_lengths[f"l={l}, t={t}"] = len(partitions)
                # Verify constraints
                assert all(anonymizer.check_k_anonimity(partition) for partition in partitions)
                assert all(anonymizer.check_l_divergence_ENTROPY(partition) for partition in partitions)
                assert all(anonymizer.check_t_closeness_TVD(partition) for partition in partitions)
                print(f"✓ All constraints satisfied for l={l}, t={t}")
        
        # Plot all distributions on one graph
        print(f"\n{'='*60}")
        print("Generating comparison plot...")
        print(f"{'='*60}\n")
        self.plot_distributions(distributions)

        # plot number of partitions for each combination of l and t
        plt.figure(figsize=(10, 6))
        plt.bar(dist_lengths.keys(), dist_lengths.values(), color='skyblue')
        plt.xlabel("Parametry (l, t)")
        plt.ylabel("Liczba grup EC")
        plt.title("Liczba grup EC dla różnych kombinacji l i t")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("liczba_grup_EC_comparison.png")
        print("✓ Saved comparison plot: liczba_grup_EC_comparison.png")


if __name__ == '__main__':
    unittest.main()
