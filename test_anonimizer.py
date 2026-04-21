import unittest
from ucimlrepo import fetch_ucirepo

from anonimize_data import MondrianAnonymizer

import pandas as pd

# wykresy
import matplotlib.pyplot as plt
from collections import Counter

class MyTestCase(unittest.TestCase):
    def test_mondrian_anonimizer(self):
        
        # 1. Import dataset
        adult = fetch_ucirepo(id=2)

        X = adult.data.features
        y = adult.data.targets

        df = pd.concat([X, y], axis=1).dropna()
        df.columns = df.columns.str.strip()
        df['income'] = df['income'].str.strip()
        
        # 2. Configure the constraints
        # k=2: At least 2 records per bucket
        # l=2: At least 2 distinct diseases per bucket
        # t=0.4: Allow 40% distribution divergence from the global baseline
        anonymizer = MondrianAnonymizer(
            data=df,
            k=5,
            l=2,
            tau=0.2,
            t=0.2,
            qi_continuous=['age'],
            qi_categorical=['sex', 'race'],
            sensitive_col='occupation'
        )


        # 3. Run Anonymization
        anonymized_df, partitions = anonymizer.run()

        # Display the grouped/generalized records
        print(anonymized_df)
        print(len(partitions))
        
        
        # --- wykres dystrybucji liczności grup ---
        group_sizes = [len(group) for group in partitions]
        size_counts = Counter(group_sizes)

        sizes = sorted(size_counts.keys())
        frequencies = [size_counts[s] for s in sizes]

        plt.figure(figsize=(20,6))

        x_labels = [str(s) for s in sizes]

        plt.bar(x_labels, frequencies)

        plt.xlabel("Rozmiar grupy EC")
        plt.ylabel("Liczba grup")
        plt.title("Histogram liczności grup wg rozmiaru")

        plt.tight_layout()
        plt.savefig("dystrybucja_licznosci_grup.png")
        
        
        print("\n--- K-Anonymity of Each Partition ---")
        print([anonymizer.k_anonimity(partition) for partition in partitions])
        assert all(anonymizer.check_k_anonimity(partition) for partition in partitions)

        print("\n--- l-divergence of Each Partition ---")
        assert all(anonymizer.check_l_divergence_ENTROPY(partition) for partition in partitions)
        
        print("\n--- t-closeness of Each Partition ---")
        assert all(anonymizer.check_t_closeness_TVD(partition) for partition in partitions)


if __name__ == '__main__':
    unittest.main()
