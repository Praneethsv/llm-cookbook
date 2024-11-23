from base_data_loader import BaseDataLoader


class TextDataLoader(BaseDataLoader):
    def __init__(self, data_loader_cfg) -> None:
        super().__init__()
        self.cfg = data_loader_cfg.text
        assert self.cfg.enabled == True

    def load_data(self):
        return super().load_data()
    
    def split(self):
        return super().split()
    
    def get_batch(self):
        return super().get_batch()
    
    def get_labels(self):
        return super().get_labels()