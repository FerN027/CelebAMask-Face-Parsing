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