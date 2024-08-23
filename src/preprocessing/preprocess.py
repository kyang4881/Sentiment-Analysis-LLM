import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import  Optional, Literal

class preprocess:
    """A class to pre-process the loaded dataset and view sentiment distribution.

    Args:
        data (pd.DataFrame): loaded dataset
        verbose (bool): whether to display visualizations
    """
    def __init__(self, data: pd.DataFrame, verbose: bool) -> None:
      self.data = data
      self.verbose = verbose

    def view_dist(self, data: Optional[pd.DataFrame] = None) -> None:
        """Obtain distribution of sentiment labels in the dataset

        Args:
            data (pd.DataFrame): optional, defaults to loaded dataset if None
        """
        if data is None:
            data = self.data
        sentiment_counts = data.groupby('sentiment')['conversation_id'].count().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        # Plot the distribution
        if self.verbose:
            print("\n\nVisualize the Chat Data Sentiment Distribution:\n\n")
            display(sentiment_counts)
            print("\n\n")
            plt.figure(figsize=(10, 5))
            plt.bar(sentiment_counts['Sentiment'], sentiment_counts['Count'])
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Distribution of Conversation IDs by Sentiment')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def remove_na(self) -> pd.DataFrame:
        """Remove records with missing data

        Returns:
            df_cleaned (pd.DataFrame): processed dataset
        """
        df_cleaned = self.data.dropna(subset=['message', 'sentiment'])
        # Calculate the number of records removed
        removed_count = len(self.data) - len(df_cleaned)
        if self.verbose:
            print(f"\n\nProcessing Data - removed {removed_count} NULL records\n\n")
        return df_cleaned

    def sampling(
        self,
        sampling_how: Optional[Literal['stratified sampling', 'random sampling', 'top n rows']] = None,
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Sample the dataset based on number of samples and method specified

        Args:
            sampling_how (str): sampling method, defaults to None
                chooose from 'stratified sampling', 'random sampling', 'top n rows', or None
            n_samples (int): number of records to sample

        Returns:
            sampled_data (pd.DataFrame): sampled dataset
        """
        # Check that n_samples is specified if sampling is required
        if sampling_how:
            assert n_samples, ("Number of records to sample should be specified via 'n_samples' argument'")

        # Perform sampling based on specifications
        if sampling_how is None:
            sampled_data = self.data

        elif sampling_how == "stratified sampling":
            unique_classes = self.data['sentiment'].unique()
            num_classes = len(unique_classes)
            samples_per_class = n_samples // num_classes

            sampled_data = pd.DataFrame(columns=self.data.columns)

            for sentiment_class in unique_classes:
                class_data = self.data[self.data['sentiment'] == sentiment_class]

                if len(class_data) < samples_per_class:
                    sample = class_data
                else:
                    sample, _ = train_test_split(
                        class_data,
                        train_size=samples_per_class,
                        stratify=class_data['sentiment'],
                        random_state=42
                    )
                sampled_data = pd.concat([sampled_data, sample]).reset_index(drop=True)

        elif sampling_how == "top n rows":
            sampled_data = self.data[:n_samples].reset_index(drop=True)

        elif sampling_how == "random sampling":
            sampled_data = self.data.sample(n=n_samples, random_state=42).reset_index(drop=True)

        else:
            raise ValueError("Unsupported sampling method. Choose from 'stratified sampling', 'random sampling', 'top n rows', or None")

        # print(f"Sampled {samples_per_class} records per sentiment class.")
        return sampled_data