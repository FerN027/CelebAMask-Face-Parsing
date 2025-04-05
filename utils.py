import numpy as np
from PIL import Image

# define palettes used in training masks
# input: torch.Size([512, 512])
# output: PIL.Image (512, 512)
def get_segmentation(mask):
    PALETTE = np.array([[i, i, i] for i in range(256)])
    PALETTE[:19] = np.array([
        [0, 0, 0], 
        [204, 0, 0], 
        [76, 153, 0], 
        [204, 204, 0], 
        [51, 51, 255], 
        [204, 0, 204], 
        [0, 255, 255], 
        [255, 204, 204], 
        [102, 51, 0], 
        [255, 0, 0], 
        [102, 204, 0], 
        [255, 255, 0], 
        [0, 0, 153], 
        [0, 0, 204], 
        [255, 51, 153], 
        [0, 204, 204], 
        [0, 51, 0], 
        [255, 153, 51], 
        [0, 204, 0]
    ])

    mask = mask.cpu().numpy()
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(PALETTE.reshape(-1).tolist())

    return mask

# measure F1-score on training/validation datasets during epoches
# input: (512, 512) numpy.array?
def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f_scores = []

    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-7)
        )

        f_scores.append(f_score)

    return np.mean(f_scores)