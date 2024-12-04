from collections import defaultdict

import numpy as np

mapping_dict = defaultdict(list)

# with open("/home/sv/Downloads/archive/tiny-imagenet-200/words.txt", "r") as f:
#     data = f.readlines()
#     for line in data:
#         wnid, label_words = line.strip().split("\t")
#         labels = [label.strip() for label in label_words.split(",")]
#         mapping_dict[wnid].append(labels)

with open("/home/sv/Downloads/tiny_imagenet/val/val_annotations.txt", "r") as f:
    data = f.readlines()
    wnids = set()
    for line in data:
        _, wnid, *box = line.strip().split("\t")
        wnids.add(wnid)

print(len(wnids))
