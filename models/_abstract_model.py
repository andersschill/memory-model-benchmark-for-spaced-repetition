import os
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, pretrain, fine_tune, output_dir=None):
        self.do_pretrain = pretrain
        self.do_fine_tune = fine_tune
        self.features = []
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def checkpoint_path(self):
        return os.path.join(self.output_dir, "model.pt")

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def pretrain(self, train_df, val_df=None):
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self, train_df, val_df=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, df):
        raise NotImplementedError

    @abstractmethod
    def nbr_params(self):
        raise NotImplementedError

    @abstractmethod
    def state_size(self):
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        raise NotImplementedError
