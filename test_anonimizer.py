import unittest

import numpy as np
from ucimlrepo import fetch_ucirepo

from anonimize_data import MondrianAnonymizer

import pandas as pd

class MyTestCase(unittest.TestCase):
    def test_mondrian_anonimizer(self):

        # 1. Import dataset
        adults_dataset = fetch_ucirepo(id=2)

        X = adults_dataset.data.features
        y = adults_dataset.data.targets

        df = pd.concat([X, y], axis=1).dropna()
        # df = adults_dataset.data.dropna()
        df.columns = df.columns.str.strip()
        df['income'] = df['income'].str.strip()
        np.random.seed(0)
        df['anon_id'] = np.random.permutation(len(df))
        # df.drop(columns=[], inplace=True)
        print(f"database columns:\n{df.columns.values}\n")

        # 2. Configure the constraints
        # k=2: At least 2 records per bucket
        # l=2: At least 2 distinct diseases per bucket
        # t=0.4: Allow 40% distribution divergence from the global baseline
        anonymizer = MondrianAnonymizer(
            data=df,
            k=10,
            l=2,
            t=0.4,
            qi_continuous=['age'],
            qi_categorical=['sex', 'race'],
            sensitive_col='occupation'
        )


        # 3. Run Anonymization
        anonymized_df, partitions = anonymizer.run()

        # Display the grouped/generalized records
        print(anonymized_df)
        print(len(partitions))
        
        print("\n--- K-Anonymity of Each Partition ---")
        print([anonymizer.check_k_anonimity(partition)[1] for partition in partitions])
        assert all(anonymizer.check_k_anonimity(partition)[0] for partition in partitions)

        print("\n--- l-divergence of Each Partition ---")
        print([anonymizer.check_l_divergence(partition)[1] for partition in partitions])
        assert all(anonymizer.check_l_divergence(partition)[0] for partition in partitions)
        
        print("\n--- t-closeness of Each Partition ---")
        print([anonymizer.check_t_closeness_TVD(partition)[1] for partition in partitions])
        assert all(anonymizer.check_t_closeness_TVD(partition)[0] for partition in partitions)

    def test_sanity(self):
        # 1. Create a mock dataset
        data = {
            'age': [23, 24, 28, 29, 34, 35, 41, 42, 45, 47],  # Continuous QI
            'salary': [50000, 52000, 60000, 61000, 75000, 76000, 90000, 92000, 95000, 98000],  # Continuous QI
            'department': ['IT', 'IT', 'HR', 'HR', 'IT', 'HR', 'Finance', 'Finance', 'IT', 'Finance'],
            # Categorical QI
            'disease': ['Flu', 'Cold', 'Flu', 'Cancer', 'Flu', 'Cold', 'Cancer', 'Flu', 'Cold', 'Cancer']
            # Sensitive
        }
        df = pd.DataFrame(data)

        # 2. Configure the constraints
        # k=2: At least 2 records per bucket
        # l=2: At least 2 distinct diseases per bucket
        # t=0.4: Allow 40% distribution divergence from the global baseline
        anonymizer = MondrianAnonymizer(
            data=df,
            k=2,
            l=2,
            t=0.4,
            qi_continuous=['age', 'salary'],
            qi_categorical=['department'],
            sensitive_col='disease'
        )

        # 3. Run Anonymization
        anonymized_df, partitions = anonymizer.run()

        # Display the grouped/generalized records
        print(anonymized_df)

        print(len(partitions))

        for idx, partition in enumerate(partitions):
            print("partition idx")

        print("\n--- K-Anonymity of Each Partition ---")
        print([anonymizer.check_k_anonimity(partition) for partition in partitions])
        assert all(anonymizer.check_k_anonimity(partition) for partition in partitions)

        print("\n--- l-divergence of Each Partition ---")
        assert all(anonymizer.check_l_divergence(partition) for partition in partitions)

        print("\n--- t-closeness of Each Partition ---")
        assert all(anonymizer.check_t_closeness_TVD(partition) for partition in partitions)


if __name__ == '__main__':
    unittest.main()
