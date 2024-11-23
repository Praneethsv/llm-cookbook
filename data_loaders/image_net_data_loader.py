from base_data_loader import BaseDataLoader
import polars as pl
import os
from cfg_loader import ConfigLoader
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch


class ImageNetDataLoader(BaseDataLoader):
    def __init__(self, data_loader_cfg) -> None:
        super().__init__()
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.data_df = None
        self.cfg = data_loader_cfg.image
        assert self.cfg.enabled == True

    def load_meta_data(self):
        """Loads Image meta data into a Data Frame
           Columns: S.no, File name, path, folder_name, 
        """
        images = os.listdir(self.cfg.data_path)
        self.data_df = pl.DataFrame({'image_names': images, 'path': self.cfg.data_path})

    def load_data(self, raw_input_path):
        " Loads Images using PIL and returns a tensor "
        image = Image.open(raw_input_path)
        image_tensor = pil_to_tensor(image)
        return image_tensor
    
    def one_hot_encoding(self, map_file):
        map_dict = {}
        with open(map_file) as f:
            for line in f:
                _, label_id, label_names = line.strip().split()
                map_dict[label_id] = label_names
        num_labels = list(map_dict.keys())
        one_hot_vectors = [0] * len(num_labels)
        return 
    
    def split(self):
        if self.data_df is None:
            self.load_meta_data()
        shuffled_data_df = self.data_df.sample(fraction=1.0, shuffle=True, seed=42)
        num_samples = shuffled_data_df.height
        train_end = int(num_samples * self.cfg.train_split)
        val_end = int(num_samples * self.cfg.val_split)
        train_df = shuffled_data_df[:train_end]
        val_df = shuffled_data_df[train_end: val_end]
        test_df = shuffled_data_df[val_end:]

        return [train_df, val_df, test_df]
    
    def preprocess(self):
        return super().preprocess()
    
    def get_batch(self, in_df: pl.DataFrame, batch_idx):
        " Takes a Data Frame, loads image data using PIL, and return tensor with batch_size images"
        # cur_idx = batch_idx
        batch_df = in_df[:batch_idx]
        images = []
        labels = []
        for image_name, path, label, _ in batch_df.iter_rows():
            image = self.load_data(path + '/' + image_name)
            images.append(image)
            labels.append(label)
        batch_images_tensor = torch.stack(images)
        labels_tensor = torch.stack(labels)
        return batch_images_tensor, labels_tensor
    
    def get_labels(self, in_df: pl.DataFrame):
        
        label_rows = []
        with open(self.cfg.annotation_file) as f:
            for line in f:
                image_names, class_label, *bbox = line.strip().split()
                label_rows.append({'image_names': image_names, 'label': class_label, 'bbox': bbox})
        
        label_data_df = pl.DataFrame(label_rows)
        merged_df = in_df.join(label_data_df, on='image_names', how='left')
        return merged_df
    

if __name__ == "__main__":
    cfg_ldr = ConfigLoader("configs", "config")
    cfg = cfg_ldr.load()
    image_net_data_loader = ImageNetDataLoader(cfg.train.data_loader)
    train_data, val_data, test_data = image_net_data_loader.split()
    res = image_net_data_loader.get_labels(test_data)
    print(res)
    image_tensor, label_tensor = image_net_data_loader.get_batch(res, 2)
    print(image_tensor)
    print(label_tensor)