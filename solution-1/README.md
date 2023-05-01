# Multi-Modal Social Media Posts Analysis

**Problem:** Let's say you are a Data Scientist working in a company that analyzes social media contents. Business team approached you and told you to build an Agent that will understand the context of a social media post that they will use to segment the content,find popularity,trends etc.

*Task*:
- Write down your approach to make the dataset, preprocess and train ML model to build
such an Intelligent Agent.
- Please note that a social media post could contain Texts, Images,Videos etc. And you
have to take all of this kind of data to a single Agent.
- You don't have to write any code. Just give us a detailed step by step description about
the process.

# Answer
Given the constraint that the social media posts may be available in different formats, I would approach the problem by leveraging **multi-modal learning**. To achieve this, I would follow the steps detailed below.

## 0. Literature Review and Prototyping
Before approaching the problem, my preliminary step would be to look for academic publications on existing or similar works. Then I would try to reproduce a few of the research works and create a prototype on a smaller scale to get an idea on the state of the art approaches. In this step I would also define my objective and modus operandi.

## 1. Dataset creation
### 1.1 Data Collection
|  |
|-------|
| Before going for the time and cost inducing methods, I would look for **existing social media datasets** on the internet. |
| After exhaustively looking for the publicly available datasets, I would look for **APIs provided by different social media websites** such as Facebook, Twitter, Reddit, Instagram, etc. |
| Finally, after merging the data retrieved by the two methods mentioned above, I would **scrape websites** that legally allow to do so. |
||

### 1.2 Data Preprocessing
I would have to deploy different data preprocessing techniques for the different data formats.

*Text* data will have to normalized, stemmed and lemmatized before being tokenized based on a vocabulary. Existing  Word2Vec models or language models may be used to extract word or sentence embeddings.

*Audio* data can be used to extract different features such as spectograms or MFCC. Some end to end deep learning approaches like the [RawNet](https://arxiv.org/abs/1904.08104) may also be explored.

*Image* data would be resized and normalized before passing them to pre-trained large CNN models to extract embeddings about them.

*Video* data has an extra temporal dimension in contrast to image data. Hence, after extracting the frames and preprocessing them, the frames will have to be passed into an RNN or LSTM model to extract a feature embedding of the video.

### 1.3 Data Imbalance

Due to the nature of the data, it is expected that data across all formats will be imbalanced. This may negatively impact the performance of the models that use this dataset. I will experiment with several methods including *over/under sampling*, *cost-sensitive learning* and *knowledge distillation* to decelerate the effects of data imbalance.

## Model Training

## Model Evaluation