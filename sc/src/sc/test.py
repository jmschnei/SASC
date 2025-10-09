
import tensorflow_datasets as tfds

info = tfds.builder("scientific_papers/arxiv").info
print(info)

# You can also query split sizes:
print("Splits available:", info.splits.keys())
for split, split_info in info.splits.items():
    print(f"{split}: {split_info.num_examples} examples")
