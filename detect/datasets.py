import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class KaistPdDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_path, split, keep_difficult=False):
        """
        :param data_path: path where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param kargs: split, one of 'visible' or 'lwir'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        
        assert self.split in {'TRAIN', 'TEST'}

        self.data_path = data_path
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_path, self.split + '_images_' + 'visible' + '.json'), 'r') as j:    # visible
            self.visible_images = json.load(j)

        with open(os.path.join(data_path, self.split + '_images_' + 'lwir' + '.json'), 'r') as j:       # lwir
            self.lwir_images = json.load(j)

        with open(os.path.join(data_path, self.split + '_objects_' + 'visible' + '.json'), 'r') as j:   # or lwir
            self.objects = json.load(j)

        assert len(self.visible_images) == len(self.lwir_images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        visible_image = Image.open(self.visible_images[i], mode='r')
        visible_image = visible_image.convert('RGB')

        lwir_image = Image.open(self.lwir_images[i], mode='r')
        lwir_image = lwir_image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4) = [(x_min, y_min, x_max, y_max), ...]
        labels = torch.LongTensor(objects['category_id'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['is_crowd'])  # (n_objects)

        # Discard difficult objects, if desired
        # Q. If we want, How deal with difficult objects? 
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        new_visible_image, new_lwir_image, boxes, labels, difficulties = transform(visible_image, lwir_image, self.split, boxes, labels, difficulties)

        return new_visible_image, new_lwir_image, boxes, labels, difficulties

    def __len__(self):
        return len(self.visible_images) # or len(self.lwir_images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists with stack.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        visible_images = list()
        lwir_images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            visible_images.append(b[0])
            lwir_images.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])

        # Combine Tensor lists to Tensor
        visible_images = torch.stack(visible_images, dim=0)
        lwir_images = torch.stack(lwir_images, dim=0)

        return visible_images, lwir_images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
