import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SewageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, transform=None, ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        random.seed(2)
        random.shuffle(self.images)
        self.len = len(self.images)
        if train:
            self.images = self.images[:int(0.7 * self.len)]
        else:
            self.images = self.images[int(0.7 * self.len):]
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item])
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path), dtype=np.float32)


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            raise ValueError("Transformer is None.")

        return image, mask


def get_loaders(image_dir,
                mask_dir,
                batch_size,
                num_worker,
                pin_memory,
                imgsize,
                train_transform=None,
                val_transform=None,
                ):
    if train_transform is None:
        train_transform = A.Compose(
            [
                A.Resize(height=imgsize[1], width=imgsize[0]),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.5330, 0.5463, 0.5493],
                    std=[0.1143, 0.1125, 0.1007],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        val_transform = A.Compose(
            [
                A.Resize(height=imgsize[1], width=imgsize[0]),
                A.Normalize(
                    mean=[0.5330, 0.5463, 0.5493],
                    std=[0.1143, 0.1125, 0.1007],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    train_dataset = SewageDataset(image_dir, mask_dir, train=True,transform=train_transform)
    val_dataset = SewageDataset(image_dir, mask_dir, train=False, transform=val_transform)


    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True,
                              pin_memory=pin_memory,
                              num_workers=num_worker, )

    val_loader = DataLoader(val_dataset,
                            batch_size,
                            shuffle=True,
                            pin_memory=pin_memory,
                            num_workers=num_worker, )

    return train_loader, val_loader
