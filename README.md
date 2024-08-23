# Sentiment Analysis with Conversational Data

---

The objectives of the sentiment analysis task are:

1. Build a multi-class sentiment analysis model based on the Topical Chat dataset from Amazon
2. Report metrics for the model
3. Write the inferencing API (reusable function/module) for the model

---

We use the end-to-end pipeline class to perform inferencing on several flan-t5 model-prompt combinations, and compare their evaluation results.

We use models from the flan-t5 family that is available to public and has a range of variants by model size. The flan-t5 model is the instruction fine-tuned version of the T5 model. As it has been fine-tuned on a mixture of tasks, it can be used for sentiment analysis by providing prompt instructions.

In addition, given the large data size and computational resources needed to perform inferencing on large number of records, we take a sample of the dataset (n=80) to serve as a test dataset for reporting performance. As the dataset is imbalanced, stratified sampling is performed to ensure that each sentiment label is represented in the test dataset.

In practice a larger data sample should be taken for testing and evaluation, however due to runtime constraints, here we demonstrate how experimentations can be done using a small dataset. The below can be replicated on a larger sample size via argument n_samples in the pipeline class.

---

## Experiment 1: Flan-T5-Small

```python

pl = pipeline(
    checkpoint = "google/flan-t5-small",
    prompt_instruction = "Which sentiment listed best describe the text below?",
    file_name = "topical_chat.csv",
    sampling_how = "stratified sampling",
    n_samples = 80,
    label_categories = ['Angry', 'Curious to dive deeper', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'],
    few_shot_examples = {
        "I love to dance a lot. How about you?": "Happy",
        "You must have ESP. I was going to tell you teh same thing! I have a sad story to tell there are less tigers living in the wild in Asia than as pets in the US.": "Sad",
        "that is so cool, and the males carry the young in that species": "Curious to dive deeper"
    },
    truncation = False,
    padding = False,
    max_length = None,
    max_new_tokens = 50,
    device = "cpu",
    eval_metrics = "all",
    plot_metrics = True
)
df_preds = pl.run()

```

### Data distribution

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/full_distribution.png" width="1200" />
</p>

### Stratified sample

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/stratified_sample.png" width="1200" />
</p>

### Few-shot examples

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/few_shot_examples.png" width="1200" />
</p>

### Prompt

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/prompt.png" width="1200" />
</p>

### Example 1 results

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/confusion_mat1.png" width="1200" />
</p>

---

## Experiment 2: Flan-T5-Base

```python

pl = pipeline(
    checkpoint = "google/flan-t5-base",
    prompt_instruction = "Which sentiment listed best describe the text below?",
    file_name = "topical_chat.csv",
    sampling_how = "stratified sampling",
    n_samples = 80,
    label_categories = ['Angry', 'Curious to dive deeper', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'],
    few_shot_examples = {
        "I love to dance a lot. How about you?": "Happy",
        "You must have ESP. I was going to tell you teh same thing! I have a sad story to tell there are less tigers living in the wild in Asia than as pets in the US.": "Sad",
        "that is so cool, and the males carry the young in that species": "Curious to dive deeper"
    },
    truncation = False,
    padding = False,
    max_length = None,
    max_new_tokens = 50,
    device = "cpu",
    eval_metrics = "all",
    plot_metrics = True,
    verbose=False
)
df_preds = pl.run()

```

### Example 2 results

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/confusion_mat2.png" width="1200" />
</p>

---

## Experiment 2: Flan-T5-Large

```python

pl = pipeline(
    checkpoint = "google/flan-t5-large",
    prompt_instruction = "Which sentiment listed best describe the text below?",
    file_name = "topical_chat.csv",
    sampling_how = "stratified sampling",
    n_samples = 80,
    label_categories = ['Angry', 'Curious to dive deeper', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'],
    few_shot_examples = {
        "I love to dance a lot. How about you?": "Happy",
        "You must have ESP. I was going to tell you teh same thing! I have a sad story to tell there are less tigers living in the wild in Asia than as pets in the US.": "Sad",
        "that is so cool, and the males carry the young in that species": "Curious to dive deeper"
    },
    truncation = False,
    padding = False,
    max_length = None,
    max_new_tokens = 50,
    device = "cpu",
    eval_metrics = "all",
    plot_metrics = True,
    verbose=False
)
df_preds = pl.run()

```

### Example 3 results

<p align="center">
  <img src="https://github.com/kyang4881/Sentiment-Analysis-with-LLMs/blob/main/docs/images/confusion_mat3.png" width="1200" />
</p>

---

As mentioned earlier, the above experiments are based on a small dataset of n=80 to demonstrate how the pipeline can be used for experimentation and performance comparison. For future work:

1. The above experiments should be replicated on a larger sample size via argument n_samples in the pipeline class, to check if similar observations are made
2. Other state of the art open-source LLMs can be tested to see if performance improvements can be obtained, such as llama-3 and mistral. (with reference to established LLM leaderboards online - HELM, Chatbot-arena)
3. The remaining non-sampled dataset can be used as a training / validation set to fine-tune the flan-t5 model (or other LLMs) for the sentiment analysis task.