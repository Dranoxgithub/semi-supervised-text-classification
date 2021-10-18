import datasets
from datasets import load_dataset
datasets_list = datasets.load_dataset("arxiv_dataset", data_dir="../raw_datasets/arxiv", ignore_verifications=True)
print(datasets_list['train'][0])