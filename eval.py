# from datasets import load_dataset
# import pandas as pd
# import matplotlib.pyplot as plt


# dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
# math500eval = dataset.to_pandas()

# # Print the DataFrame
# print(math500eval)

from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
print(dataset[0])
