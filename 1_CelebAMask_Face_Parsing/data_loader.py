import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

train_images_dir = 'data/dev-public/train/images'
train_masks_dir = 'data/dev-public/train/masks'
test_images_dir = 'data/test-public/test/images'

# TODO:
# carefully define transforms for images/masks
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def squeeze_first_dimension(x):
    # squeeze torch.Size([1, 512, 512]) into .Size([512, 512])
    return x.squeeze(0)

def to_long(x):
    return x.long()

mask_transforms = transforms.Compose([
    transforms.PILToTensor(),
    squeeze_first_dimension,
    to_long

    # lambda x: x.squeeze(0),
    # lambda x: x.long()

    # transforms.Lambda(
    #     lambda x: x.squeeze(0)
    # )
])

class FaceParsingDataset(Dataset):
    def __init__(self, is_train, train_images_dir=None, train_masks_dir=None, test_images_dir=None, image_transform=None, mask_transform=None):
        """
        Args:
            is_train (bool): True for training (expects images and masks), False for testing (expects only images).
            train_images_dir (str, optional): Directory with training images.
            train_masks_dir (str, optional): Directory with training masks.
            test_images_dir (str, optional): Directory with test images (no masks).
            image_transform (callable, optional): Transformations for images.
            mask_transform (callable, optional): Transformations for masks.
        """
        self.is_train = is_train
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if self.is_train:
            assert train_images_dir is not None and train_masks_dir is not None, \
                "For training, both train_images_dir and train_masks_dir must be provided."
            self.images_dir = train_images_dir
            self.masks_dir = train_masks_dir
            self.image_files = sorted(os.listdir(train_images_dir))
            self.mask_files = sorted(os.listdir(train_masks_dir))
            assert len(self.image_files) == len(self.mask_files), "Mismatch between the number of images and masks."
        else:
            assert test_images_dir is not None, "For testing, test_images_dir must be provided."
            self.images_dir = test_images_dir
            self.image_files = sorted(os.listdir(test_images_dir))
            self.masks_dir = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image.
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # If in training mode, load the corresponding mask.
        if self.is_train:
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
            mask = Image.open(mask_path)
            if self.mask_transform:
                mask = self.mask_transform(mask)

        # Else put in a dummy mask
        else:
            mask = torch.zeros((1, 512, 512), dtype=torch.int64)

        return image, mask

def get_dataloader(is_train: bool):
    """
    Returns the DataLoader for the specified mode (train/test).
    """
    if is_train:
        dataset = FaceParsingDataset(
            is_train=True,
            train_images_dir=train_images_dir,
            train_masks_dir=train_masks_dir,
            image_transform=image_transforms,
            mask_transform=mask_transforms
        )

        return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    else:
        dataset = FaceParsingDataset(
            is_train=False,
            test_images_dir=test_images_dir,
            image_transform=image_transforms
        )

        return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
