# Multi-Modal Social Media Posts Analysis

Given the constraint that the social media posts may be available in different formats, I would approach the problem by leveraging **multi-modal learning**.

## 0. Literature Review and Prototyping
Before approaching the problem, my preliminary step will be to look for academic publications on existing or similar works. Then I will try to reproduce a few of the research works and create a prototype on a smaller scale to get an idea on the state of the art approaches. In this step I would also define my objective and modus operandi.

## 1. Dataset creation
### 1.1 Data Collection
|  |
|-------|
| Before going for the time and cost inducing methods, I will look for **existing social media datasets** on the internet. |
| After exhaustively looking for the publicly available datasets, I would look for **APIs provided by different social media websites** such as Facebook, Twitter, Reddit, Instagram, etc. |
| Finally, after merging the data retrieved by the two methods mentioned above, I will **scrape websites** that legally allow to do so. |
||

### 1.3 Data Imbalance

Due to the nature of the data, it is expected that data across all formats will be imbalanced. This may negatively impact the performance of the models that use this dataset. I will experiment with several methods including over/under sampling, cost-sensitive learning and knowledge distillation to decelerate the effects of data imbalance.


