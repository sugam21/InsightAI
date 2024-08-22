import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import Tensor


class DataTransform:
    """Data transformation pipeline
    base:
        training images without major alteration
    train:
        training images with major alteration
    valid:
        validation and testing images alteration
        """
    def __init__(self, input_size: int):
        self.data_transform = {
            "base": A.Compose(
                [
                    A.Resize(width=input_size, height=input_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(normalization="min_max_per_channel"),
                    ToTensorV2(),
                ]
            ),
            "train": A.Compose(
                [

                    A.Resize(width=input_size, height=input_size),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.MultiplicativeNoise(multiplier=[0.5, 2], per_channel=True, p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(0, 0.1), contrast_limit=(0.01, 0.1), p=1.0),
                    A.Normalize(normalization="min_max_per_channel"),
                    ToTensorV2(),
                ]
            ),
            "valid": A.Compose(
                [
                    A.Resize(width=input_size, height=input_size),
                    A.Normalize(normalization="min_max_per_channel"),
                    ToTensorV2(),
                ]
            ),
        }

    def __call__(self, image: Tensor, is_train: bool = False, is_augment: bool = False):
        transform_type: str
        if is_train:
            transform_type = "train" if is_augment else "base"
        else:
            transform_type = "valid"
        transformed = self.data_transform[transform_type](image=image)
        return transformed['image']

