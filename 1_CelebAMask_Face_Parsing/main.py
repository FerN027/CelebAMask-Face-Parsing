import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from data_loader import get_dataloader



if __name__ == "__main__":
    train_loader = get_dataloader(is_train=True)
    imgs, _ = next(iter(train_loader))
    