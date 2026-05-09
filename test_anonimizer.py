import unittest

from anonimize_data import MondrianAnonymizer

import pandas as pd

class MyTestCase(unittest.TestCase):
    def test_mondrian_anonimizer(self):
        # 1. Create a mock dataset
        data = {
            'age': [23, 24, 28, 29, 34, 35, 41, 42, 45, 47],  # Continuous QI
            'salary': [50000, 52000, 60000, 61000, 75000, 76000, 90000, 92000, 95000, 98000],  # Continuous QI
            'department': ['IT', 'IT', 'HR', 'HR', 'IT', 'HR', 'Finance', 'Finance', 'IT', 'Finance'],  # Categorical QI
            'disease': ['Flu', 'Cold', 'Flu', 'Cancer', 'Flu', 'Cold', 'Cancer', 'Flu', 'Cold', 'Cancer']  # Sensitive
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
        anonymized_df = anonymizer.run()

        # Display the grouped/generalized records
        print(anonymized_df)
        print([partition for partition in anonymized_df])
        print("\n--- K-Anonymity of Each Partition ---")
        print([anonymizer.k_anonimity(partition) for partition in anonymized_df])
        assert all(anonymizer.check_k_anonimity(partition) for partition in anonymized_df)


if __name__ == '__main__':
    unittest.main()
