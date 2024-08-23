import time
import random
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List

class inference:
    """A class for making inferences using a given flan-t5 model.

    Args:
        data (pd.DataFrame): processed dataset to perform inferencing on
        input_col (str): name of column in data containing text messages as input
        label_categories (list(str)): list of sentiment categories to classify message
        checkpoint (str): name of model and tokenizer checkpoint for inferencing, e.g. "google/flan-t5-small"
        prompt_instruction (str): prompt instruction text
        few_shot_examples (dict): optional, examples to use for few-shot learning
            provided in {<message1>: <label1>, <message2>: <label2>, ...} format
        truncation (bool): whether to truncate input
        padding (bool): whether to apply padding to inputs
        max_length (int/None): argument that controls the length of the padding and truncation of inputs
            If None, defaults to the maximum input length the model can accept
        max_new_tokens (int): maximum number of tokens to generate
        device (str): whether to use cpu or gpu device
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_col: str = "message",
        label_categories: List[str] = ['Angry', 'Curious to dive deeper', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'],
        checkpoint: str = "google/flan-t5-base",
        prompt_instruction: str = "Which sentiment listed best describes the text below?",
        few_shot_examples: Dict[str, None] = {},
        truncation: bool = False,
        padding: bool = False,
        max_length: int = None,
        max_new_tokens: int = 5,
        device: str = "cpu"
    ) -> None:

        self.data = data
        self.input_col = input_col
        self.label_categories = label_categories
        self.checkpoint = checkpoint
        self.prompt_instruction = prompt_instruction
        self.few_shot_examples = few_shot_examples
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device

        # Instantiate model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def process_prediction(self, pred: str) -> str:
        """Apply text processing on sentiment prediction from model.

        Args:
            pred (str): model inferencing output

        Returns:
            pred (str): processed prediction
        """
        # Remove punctuations
        pred = re.sub(r'[^\w\s]', '', pred)
        # Convert to lower case
        pred = pred.lower()
        # Strip white space from the left and right
        pred = pred.strip()
        # Ensure there's at most 1 space between words
        pred = re.sub(r'\s+', ' ', pred)

        # Check if label_categories are a substring of pred
        for category in self.label_categories:
            if category.lower() in pred:
                return category.lower()
        return pred

    def generate_results(self) -> pd.DataFrame:
        """Generate inferences from the model on the dataset.

        Returns:
            (pd.DataFrame): dataset with additional columns for prompt and inference results
        """
        start_time = time.time()
        options = "\n".join(f"-{x}" for x in sorted(self.label_categories))
        label_options = f"\n{options}\n\nA:" if options != "" else options

        preds = []
        preds_original = []
        prompts = []
        i = 0
        counter = 0
        while i < len(self.data):

            # Format prompt
            example_items = ''.join([f'Q: {self.prompt_instruction}\n\nText: {k}\n\nOPTIONS:{label_options} {v}\n\n' for k, v in self.few_shot_examples.items()])
            examples = f"{example_items}" if example_items else ""
            prompt = f"""\n\n{examples}\n\nQ: {self.prompt_instruction}\n\nText: {self.data[self.input_col][i]}\n\nOPTIONS:{label_options}"""

            # Tokenize inputs and obtain inference output
            encoded_inputs = self.tokenizer(prompt, padding=self.padding, max_length=self.max_length, truncation=self.truncation, return_tensors="pt").to(self.device)
            encoded_outputs = self.model.generate(**encoded_inputs, max_new_tokens= self.max_new_tokens)
            decoded_outputs = self.tokenizer.batch_decode(encoded_outputs, skip_special_tokens=True)[0]
            prompts.append(prompt)
            #print(prompt)

            # Process inference output
            if self.process_prediction(decoded_outputs) in [c.lower() for c in self.label_categories]:
                preds.append(self.process_prediction(decoded_outputs))
                preds_original.append(decoded_outputs)
                i += 1
            # If categories are not returned, try again
            elif counter <3:
                counter += 1
                continue
            # If categories are not returned after multiple tries, randomly return a category
            else:
                preds_random_choice = random.choice(self.label_categories)
                preds.append(preds_random_choice)
                preds_original.append("NULL")
                i += 1
                counter = 0

        end_time = time.time()
        print(f"\nTotal time taken: {(end_time-start_time)/60:.2f} minutes")
        self.data['preds'] = preds
        self.data['preds_original'] = preds_original
        self.data['prompts'] = prompts
        print(self.data['prompts'][0], "\n\n")

        return self.data