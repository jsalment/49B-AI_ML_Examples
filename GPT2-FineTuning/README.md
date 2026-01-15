# 10617-reddit-text-generation

There are 3 files for running our experiments:

1. Reddit_Dataset_Prep.ipynb
Run this first to prepare your train/test datasets. It will first load a subreddit from ConvoKit,
and then you can modify how the train dataset is sorted. Make sure that the dataset is split into train/test
before performing sorting within the test set itself. Finally it will save the prepared dataset
on disk in the `Datasets` object.

2. Reddit_Fine_Tuning.ipynb
You'll run this next to fine-tune either GPT2 or your existing fine-tuned model. The latter is very useful
as you would normally want to train with fewer epochs at a time (I do 1-3), and then
just iteratively fine-tune it with further epochs, in case anything happens
where the kernel crashes, Sagemaker logs you out, etc. Each epoch takes around 3 hours.

3. Model Evaluation.ipynb
This is where you'll evaluate the BLEU scores of your model on the test set. 
