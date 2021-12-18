import glob
from typing import Tuple
import os.path as osp

from PIL import Image
import torch.utils.data as data
from torchvision import transforms


def make_datapath_list(phase="train"):
    root_path = osp.join(".","data","hymenoptera_data")
    target_path = osp.join(root_path,phase,"**","*.jpg")
    print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class HymenopteraDataset(data.Dataset):
    """
    Dataset for ants and bees
    """
    def __init__(self, file_list, transform=None, phase="train") -> None:
        super().__init__()
        self._file_list = file_list
        self._transform = transform
        self._phase = phase

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, index):
        img_path = self._file_list[index]
        img = Image.open(img_path)  # (height, width, channel)

        img_transformed = self._transform(img, self._phase) # (channel, height, width)
        if self._phase == "train":
            label = img_path[30:34]
        elif self._phase == "val":
            label = img_path[28:32]

        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1
        
        return img_transformed, label


class ImageTransform():
    def __init__(
        self,
        resize: int=224,
        mean: Tuple[float, float, float]=(0.485, 0.456, 0.406),
        std: Tuple[float, float, float]=(0.229, 0.224, 0.225)
    ) -> None:
        self._transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)
                ),  # 원본 영상의 0.5 ~ 1 사이의 크기로 영상을 잘라내서 resize 로 변환
                transforms.RandomHorizontalFlip(),  # 임의 비율로 이미지의 상하 반전
                transforms.ToTensor(),  # PIL image를 Tensor로 변환
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize), # 이미 영상을 resize로 크기 조정했는데, center crop하는게 의미 있을까?
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        preprocess image and return result.
        
        :param img: image for preprocess
        :param phase: phase of model development process(train, val)
        """
        return self._transform[phase](img)