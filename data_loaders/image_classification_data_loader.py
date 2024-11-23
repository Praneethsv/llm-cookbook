from data_loaders.image_net_data_loader import ImageDataLoader


class ImageClassificationDataLoader(ImageDataLoader):
    def __init__(self, data_loader_cfg) -> None:
        super().__init__(data_loader_cfg)

    def get_labels(self):
        return super().get_labels()