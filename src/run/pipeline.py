import pandas as pd
from typing import Dict, List, Union
import evaluation.evaluate as evaluate
from inference.predict import inference

class pipeline:
    """A class for end-to-end pipeline to enable quicker experimentation

    Args:
        checkpoint (str): name of model and tokenizer checkpoint for inferencing, e.g. "google/flan-t5-small"
        prompt_instruction (str): prompt instruction text
        file_name (str): name of dataset file
        file_path (str): path containing dataset
        sampling_how (str): sampling method
            chooose from 'full sample', 'stratified sampling', 'random sampling', or 'top n rows'
        n_samples (int): number of records to sample
        input_col (str): name of column in data containing text messages as input
        label_col (str): name of column in data containing true sentiment labels
        label_categories (list(str)): list of sentiment categories to classify message
        few_shot_examples (dict): examples to use for few-shot learning
            provided in {<message1>: <label1>, <message2>: <label2>, ...} format
        truncation (bool): whether to truncate input
        padding (bool): whether to apply padding to inputs
        max_length (int/None): argument that controls the length of the padding and truncation of inputs
            If None, defaults to the maximum input length the model can accept
        max_new_tokens (int): maximum number of tokens to generate
        device (str): whether to use cpu or gpu device
        eval_metrics (str/list(str)): evaluation metrics to compute
            choose from 'accuracy', 'f1', 'precision', or 'recall'
        plot_metrics (bool): whether to plot confusion matrix
        verbose (bool): whether to display preprocessing visualizations
    """
    def __init__(
        self,
        checkpoint: str,
        prompt_instruction: str,
        file_name: str,
        file_path: str = "./",
        sampling_how: Union[str, None] = None,
        n_samples: Union[int, None] = None,
        input_col: str = "message",
        label_col: str = "sentiment",
        label_categories: List[str] = ['Angry', 'Curious to dive deeper', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'],
        few_shot_examples: Dict = {},
        truncation: bool = False,
        padding: bool = False,
        max_length: int = None,
        max_new_tokens: int = 50,
        device: str = "cpu",
        eval_metrics: Union[str, List[str]] = "all",
        plot_metrics: bool = True,
        verbose: bool = True
    ) -> None:

        self.checkpoint = checkpoint
        self.prompt_instruction = prompt_instruction
        self.file_name = file_name
        self.file_path = file_path
        self.sampling_how = sampling_how
        self.n_samples = n_samples
        self.input_col = input_col
        self.label_col = label_col
        self.label_categories = label_categories
        self.few_shot_examples = few_shot_examples
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.eval_metrics = eval_metrics
        self.plot_metrics = plot_metrics
        self.verbose = verbose

    def run(self) -> pd.DataFrame:
        """Run pipeline to import and process data, and obtain inferencing and evaluation results

        Returns:
            df_preds (pd.DataFrame): dataset containing the sentiment predictions
        """
        data = importData(
            file_name=self.file_name,
            file_path=self.file_path
        ).get_file()

        # Import and preprocess data
        preprocessor = preprocess(data=data, verbose=self.verbose)
        preprocessor.view_dist(data=data)
        data = preprocessor.remove_na()
        data = preprocessor.sampling(
            n_samples=self.n_samples,
            sampling_how=self.sampling_how
        )
        if self.sampling_how:
            print(f"\n\nSampling Method Selected: {self.sampling_how}\n\n")
        preprocessor.view_dist(data=data)

        # Perform inference
        infer = inference(
            data = data,
            input_col = self.input_col,
            label_categories = self.label_categories,
            checkpoint = self.checkpoint,
            prompt_instruction = self.prompt_instruction,
            few_shot_examples = self.few_shot_examples,
            truncation = self.truncation,
            padding = self.padding,
            max_length = self.max_length,
            max_new_tokens = self.max_new_tokens,
            device = self.device
        )
        df_preds = infer.generate_results()

        # Evaluate results
        eval = evaluate(
            df_preds = df_preds,
            eval_metrics = self.eval_metrics,
            label_col = self.label_col,
            label_categories = self.label_categories,
            plot_metrics = self.plot_metrics
        )
        eval.score()

        return df_preds