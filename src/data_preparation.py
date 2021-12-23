# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyMNIST, MyCIFAR10
from typing import Optional, Callable
import torch


#class
class MyMNIST(MyMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        normal_class = '0 - zero'
        # the index of normal is 0 and abnormal is 1
        #TODO: needs to fix the problem that set class_to_idx
        self.class_to_idx = {v: 1 for v in self.class_to_idx.keys()}
        self.class_to_idx[normal_class] = 0
        self.classes = ['normal', 'abnormal']
        if train:
            self.data = self.data[self.targets == int(normal_class[0])]
            self.targets = self.targets[self.targets == int(normal_class[0])]
        else:
            self.targets = torch.where(self.targets == int(normal_class[0]), 0,
                                       1)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    #
    train_dataset = MyMNIST(root='data/MNIST/',
                            train=True,
                            transform=None,
                            target_transform=None,
                            download=False)
