import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List, Union

class evaluate:
    """A class for evaluating the predicted sentiment labels.

    Args:
        df_preds (pd.DataFrame): dataset obtained after running inference class, containing the sentiment predictions
        eval_metrics (str/list(str)): evaluation metrics to compute
            choose from 'accuracy', 'f1', 'precision', or 'recall'
        label_col (str): name of column in data containing true sentiment labels
        label_categories (list(str)): list of sentiment categories to classify message
        plot_metrics (bool): whether to plot confusion matrix
    """
    def __init__(
        self,
        df_preds: pd.DataFrame,
        eval_metrics: Union[str, List[str]],
        label_col: str = "sentiment",
        label_categories: List[str] = ['Angry', 'Curious to dive deeper', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'],
        plot_metrics: bool = False
    ) -> None:

        self.preds = df_preds.copy()['preds']
        self.true_val = df_preds.copy()[label_col].apply(lambda x: x.lower()) # convert to lower case for comparison
        self.label_col = label_col
        self.label_categories = label_categories
        self.eval_metrics = eval_metrics
        self.plot_metrics = plot_metrics
        self.results = {}

    def score(self) -> Dict[str, float]:
        """Obtain evaluation metrics

        Returns:
            self.results (dict): dictionary of evaluation metrics
        """
        if 'all' in self.eval_metrics or 'accuracy' in self.eval_metrics:
            self.results['accuracy'] = accuracy_score(self.true_val, self.preds)
        if 'all' in self.eval_metrics or 'f1' in self.eval_metrics:
            self.results['f1'] = f1_score(self.true_val, self.preds, average='weighted')
        if 'all' in self.eval_metrics or 'precision' in self.eval_metrics:
            self.results['precision'] = precision_score(self.true_val, self.preds, average='weighted', zero_division=0)
        if 'all' in self.eval_metrics or 'recall' in self.eval_metrics:
            self.results['recall'] = recall_score(self.true_val, self.preds, average='weighted', zero_division=0)

        if self.plot_metrics:
            cm = confusion_matrix(self.true_val, self.preds)
            plt.figure(figsize=(8, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=self.label_categories, yticklabels=self.label_categories)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()

        print("\nEvaluation Metrics:\n", self.results)

        return self.results