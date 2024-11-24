import os
import pickle
from cfg_loader import ConfigLoader
import numpy as np
import polars as pl
import torch
from base_data_loader import BaseDataLoader


class CifarDataLoader(BaseDataLoader):
    def __init__(self, data_loader_cfg):
        super().__init__()
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.data_df = None
        self.cfg = data_loader_cfg.image
        self.num_labels = self.get_num_labels()
        assert self.cfg.enabled == True

    def get_num_labels(self):

        with open(self.cfg.annotation_map_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        return len(data["label_names"])

    def load_data(self):

        files = os.listdir(self.cfg.data_path)
        images = []
        labels = []
        names = []

        for file in files:
            if "batch" in file.strip("_"):
                with open(self.cfg.data_path, "rb") as f:
                    batch_label, classes, data, filenames = pickle.load(
                        f, encoding="latin1"
                    )

                for filename, data, class_ in zip(filenames, data, classes):
                    one_hot_labels = [0] * self.num_labels
                    array_data = torch.tensor(data).view(3, 32, 32).permute(1, 2, 0)
                    images.append(array_data)
                    one_hot_labels[class_] = 1
                    labels.append(one_hot_labels)
                    names.append(filename)
        self.data_df = pl.DataFrame(
            {"names": names, "images": images, "labels": labels}
        )

    def get_labels(self):
        super().get_labels()

    def preprocess(self):
        return super().preprocess()

    def split(self):
        if self.data_df is None:
            self.load_data()
        shuffled_data_df = self.data_df.sample(fraction=1.0, shuffle=True, seed=42)
        num_samples = shuffled_data_df.height
        train_end = int(num_samples * self.cfg.train_split)
        val_end = int(num_samples * self.cfg.val_split)
        train_df = shuffled_data_df[:train_end]
        val_df = shuffled_data_df[train_end:val_end]
        test_df = shuffled_data_df[val_end:]

        return [train_df, val_df, test_df]

    def get_batch(self, in_df: pl.DataFrame, batch_idx):
        # cur_idx = batch_idx
        batch_df = in_df[:batch_idx]
        images = []
        labels = []
        for file_name, data, label in batch_df.iter_rows():
            images.append(data)
            labels.append(label)
        batch_images_tensor = torch.stack(images)
        labels_tensor = torch.stack(labels)
        return batch_images_tensor, labels_tensor


if __name__ == "__main__":
    cfg_ldr = ConfigLoader("configs", "config")
    cfg = cfg_ldr.load()
    cifar_data_loader = CifarDataLoader(cfg.train.data_loader)
    train_data, val_data, test_data = cifar_data_loader.split()
    res = cifar_data_loader.get_labels(test_data)
    print(res)
    image_tensor, label_tensor = cifar_data_loader.get_batch(res, 2)
    print(image_tensor)
    print(label_tensor)
