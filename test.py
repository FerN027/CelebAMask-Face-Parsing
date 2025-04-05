import os
import torch

from PIL import Image

from model import EnhancedFaceParsingNet
from data_loader import get_dataloader
from utils import get_segmentation

def make_predictions(
    device,
    learned_model_path="model.pth",      
    output_dir="data/test-public/test/predicted_masks"
):
    # Load the trained model    
    model = EnhancedFaceParsingNet()
    model.load_state_dict(torch.load(learned_model_path))
    model = model.to(device)

    # get test dataloader and file names
    test_loader, names = get_dataloader(is_train=False)

    ready_for_output = []

    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)  # [B, 19, 512, 512]
            preds = torch.argmax(logits, dim=1)  # [B, 512, 512]

            for i in range(preds.shape[0]):
                pred = preds[i]  # [512, 512]
                mask = get_segmentation(pred)

                ready_for_output.append(mask)

    # Save all predicted masks
    assert len(ready_for_output) == len(names), "Mismatch in number of predictions and names"

    for i in range(len(ready_for_output)):
        ready_for_output[i].save(os.path.join(output_dir, names[i]))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    make_predictions(device)
    print("Predictions saved successfully.")
