import kornia.augmentation as K
import polars as pl
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from data_loaders.base_data_loader import BaseDataLoader


class ImageNetDataLoader(BaseDataLoader):
    def __init__(self, data_loader_cfg, device="cpu") -> None:
        super().__init__()
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.data_df = None
        self.cfg = data_loader_cfg.image
        self.device = device
        assert self.cfg.enabled == True

    def load_meta_data(self):
        """Loads Image meta data into a Data Frame
        Columns: S.no, File name, path, folder_name,
        """
        map_dict = self.get_labels()
        with open(self.cfg.annotation_file) as f:
            images = []
            bboxes = []
            annotations = []
            for line in f:
                labels = [0] * 200
                image_names, class_label, *bbox = line.strip().split("\t")
                labels[map_dict[class_label]] = 1
                images.append(image_names)
                annotations.append(labels)
                bboxes.append(bbox)

        self.data_df = pl.DataFrame(
            {
                "image_names": images,
                "path": self.cfg.data_path,
                "labels": annotations,
                "bboxes": bboxes,
            }
        )

    def load_data(self, raw_input_path):
        "Loads Images using PIL and returns a tensor"
        image = Image.open(raw_input_path)
        image = image.convert("RGB") if image.mode == "L" else image
        image_tensor = pil_to_tensor(image).float() / 255.0
        return image_tensor.to(self.device)

    def split(self):
        if self.data_df is None:
            self.load_meta_data()
        shuffled_data_df = self.data_df.sample(fraction=1.0, shuffle=True, seed=42)
        num_samples = shuffled_data_df.height
        train_end = int(num_samples * self.cfg.train_split)
        val_end = int(num_samples * self.cfg.val_split)
        train_df = shuffled_data_df[:train_end]
        val_df = shuffled_data_df[train_end:val_end]
        test_df = shuffled_data_df[val_end:]

        return [train_df, val_df, test_df]

    def preprocess(self, image_batch: torch.Tensor):
        aug = K.AugmentationSequential(
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30.0, 50.0], p=1.0),
        )
        out_tensors = aug(image_batch)
        return out_tensors.to(self.device)

    def get_batch(self, in_df: pl.DataFrame, batch_idx):
        "Takes a Data Frame, loads image data using PIL, and return tensor with batch_size images"
        batch_df = in_df[:batch_idx]
        images = []
        labels = []
        for image_name, path, label, _ in batch_df.iter_rows():
            image = self.load_data(path + "/" + image_name)
            images.append(image)
            labels.append(torch.tensor(label, dtype=torch.float32))
        batch_images_tensor = (
            self.preprocess(torch.stack(images))
            if self.cfg.preprocess
            else torch.stack(images).to(self.device)
        )
        batch_labels_tensor = torch.stack(labels).to(self.device)
        return batch_images_tensor, batch_labels_tensor

    def get_labels(self):
        map_dict = {}
        with open(self.cfg.annotation_map_file) as meta_file:
            data = meta_file.readlines()
            for i, line in enumerate(data):
                map_dict[line.strip()] = i
        return map_dict
