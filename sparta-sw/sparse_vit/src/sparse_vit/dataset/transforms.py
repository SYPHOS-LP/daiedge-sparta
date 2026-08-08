from typing import Callable

import torch

from torchvision.transforms import v2


def _img_transform_basic():
    """
    Get basic image transformation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224), interpolation=3, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return trns


def _img_transform_augment_deit(img_size: int = 224):
    """
    Image augmentations applied in `DeiT III: Revenge of the ViT`.

    Augmentations on `DeiT III: Revenge of the ViT`
    ```
    https://github.com/facebookresearch/deit/blob/main/augment.py#L90
    ```

    ```https://pillow.readthedocs.io/en/stable/reference/Image.html#resampling-filters```

    NOTE:
    InterpolationMode.NEAREST:  0
    InterpolationMode.NEAREST_EXACT:
    InterpolationMode.BILINEAR:  2
    InterpolationMode.BICUBIC:  3
    InterpolationMode.LANCZOS:  1
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trns = v2.Compose(
        [
            v2.ToImage(),  # convert to `Image` class
            v2.Resize(size=(img_size, img_size), interpolation=3, antialias=True),
            # NOTE: Disabling `RandomCrop` as inputs are square
            # Maybe replace with `RandomResizedCrop`
            # v2.RandomCrop(size=(img_size, img_size), padding=4, padding_mode="reflect"),
            v2.RandomResizedCrop(
                scale=(0.6, 1.0), size=(img_size, img_size), ratio=(1.0, 1.0)
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.RandomGrayscale(p=1.0),
                    v2.RandomSolarize(128, p=1.0),
                    v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                ]
            ),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return trns


def _img_transform_augment():
    """
    Get image augmentation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.2),
        ]
    )

    return trns


def _img_transform_basic_dct():
    """
    Get basic image transformation.
    """
    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224), interpolation=3, antialias=True),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    return trns


def _img_transform_augment_deit_dct(img_size: int = 224):
    """
    Image augmentations applied in `DeiT III: Revenge of the ViT`.

    Augmentations on `DeiT III: Revenge of the ViT`
    ```
    https://github.com/facebookresearch/deit/blob/main/augment.py#L90
    ```

    ```https://pillow.readthedocs.io/en/stable/reference/Image.html#resampling-filters```

    NOTE:
    InterpolationMode.NEAREST:  0
    InterpolationMode.NEAREST_EXACT:
    InterpolationMode.BILINEAR:  2
    InterpolationMode.BICUBIC:  3
    InterpolationMode.LANCZOS:  1
    """
    trns = v2.Compose(
        [
            v2.ToImage(),  # convert to `Image` class
            v2.Resize(size=(img_size, img_size), interpolation=3, antialias=True),
            # NOTE: Disabling `RandomCrop` as inputs are square
            # Maybe replace with `RandomResizedCrop`
            # v2.RandomCrop(size=(img_size, img_size), padding=4, padding_mode="reflect"),
            v2.RandomResizedCrop(
                scale=(0.6, 1.0), size=(img_size, img_size), ratio=(1.0, 1.0)
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.RandomGrayscale(p=1.0),
                    v2.RandomSolarize(128, p=1.0),
                    v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                ]
            ),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    return trns


def _img_transform_augment_dct():
    """
    Get image augmentation.
    """
    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    return trns


def _img_imagenet_transform_basic():
    """
    Get basic image transformation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=256, interpolation=3, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return trns


def _img_imagenet_transform_augment(img_size: int = 224):
    """
    Get image augmentation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=img_size, interpolation=3, antialias=True),
            v2.RandomCrop(size=(img_size, img_size), padding=4, padding_mode="reflect"),
            # v2.RandomResizedCrop(
            #     size=224,
            #     scale=(0.08, 1.0),
            #     ratio=(0.75, 1.3333333333333333),
            #     interpolation=v2.InterpolationMode.BICUBIC,
            # ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.2),
        ]
    )

    return trns


def _img_imagenet_transform_basic_dct():
    """
    Get basic image transformation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=256, interpolation=3, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    return trns


def _img_imagenet_transform_augment_dct():
    """
    Get image augmentation.
    """
    trns = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=224,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    return trns


def apply_transform_factory(transform: Callable, batched: bool = True) -> Callable:

    trns = transform()

    def _apply_transform_per_batch(batch):
        """
        Apply image augmentation for each item of input batch.

        NOTE: This function is defined for a dataset `.set_transform` method.
        """
        batch["image"] = [trns(img.convert("RGB")) for img in batch["image"]]

        return batch

    def _apply_transform_per_sample(sample):
        """
        Apply image augmentation on sample.

        NOTE: This function is defined for a dataset `.set_transform` method.
        """
        sample["image"] = trns(sample["image"].convert("RGB"))

        return sample

    if batched is True:
        return _apply_transform_per_batch
    else:
        return _apply_transform_per_sample


def cutmix_or_mixup(num_classes: int):
    """
    Cutmix or mixup
    """
    # Use same parameters as DeiTIII on ImageNet
    cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
    mixup = v2.MixUp(alpha=0.8, num_classes=num_classes)

    return v2.RandomChoice([cutmix, mixup])
