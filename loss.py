import torch
import torch.nn as nn

from data_loader import FaceParsingDataset, train_images_dir, train_masks_dir, image_transforms, mask_transforms

def my_criterion(option, device):
    def compute_class_weights(train_dataset):  # Compute class weights for imbalanced face parsing dataset
        class_pixels = torch.zeros(19)
        total_pixels = 0

        # Count pixels per class across all training images
        for _, mask in train_dataset:  # Assuming Dataset.__getitem__ returns (image, mask)
            mask_counts = torch.bincount(mask.flatten(), minlength=19)
            class_pixels += mask_counts
            total_pixels += mask.numel()

        # Calculate weights using median frequency balancing
        frequencies = class_pixels / total_pixels
        median_freq = torch.median(frequencies)
        weights = median_freq / frequencies

        print("Class frequencies:", frequencies)
        print("Class weights:", weights)

        return weights.to(device)

    # per-pixel cross entropy loss
    if option == "CE":
        return nn.CrossEntropyLoss()

    # CE with weights for different classes
    elif option == "CE_W":
        train_dataset = FaceParsingDataset(
            is_train=True,
            train_images_dir=train_images_dir,
            train_masks_dir=train_masks_dir,
            image_transform=image_transforms,
            mask_transform=mask_transforms
        )

        return nn.CrossEntropyLoss(weight=compute_class_weights(train_dataset))

    else:
        raise ValueError("Error: not match any valid option!")
