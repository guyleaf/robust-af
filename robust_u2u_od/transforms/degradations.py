import random
from typing import Union

import albumentations as A
import cv2
import imgaug.augmenters as iaa
import numpy as np

# TODO: better organization (class instead of functional)


def _apply_albu_transforms(transforms: A.BaseCompose, *args, **kwargs):
    # share the global random state with albumentations to keep reprociblility
    # albumentations: https://albumentations.ai/docs/faq/?h=seed#how-to-have-reproducible-augmentations_1
    # numpy: https://numpy.org/doc/1.26/release/1.24.0-notes.html#the-bit-generator-underlying-the-singleton-randomstate-can-be-changed
    np_generator = np.random.default_rng(np.random.get_bit_generator())
    py_random = random.Random()
    py_random.setstate(random.getstate())
    transforms.set_random_state(np_generator, py_random)

    results = transforms(*args, **kwargs)

    # sync the random state from albumentations to global
    # in numpy, the underlying bit generator uses the shared global random state. so, we don't need to sync it.
    random.setstate(py_random.getstate())
    return results


def apply_snow(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.RandomSnow(brightness_coeff=1.0, snow_point_range=(0.3, 0.7), p=1)]
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]

    aug = iaa.Snowflakes(
        density=0.35, flake_size=(0.6, 0.8), speed=(0.01, 0.015), angle=0
    )
    image = aug.augment_image(image)
    return image


def apply_fog(image: np.ndarray, **kwargs):
    aug = iaa.Fog()
    image = aug.augment_image(image)

    transforms = A.Compose(
        [A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.08, p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_rain(image: np.ndarray, **kwargs):
    aug = iaa.Rain(drop_size=(0.40, 0.50), speed=(0.05, 0.1))
    image = aug.augment_image(image)
    return image


def apply_gaussian_noise(image: np.ndarray, **kwargs):
    # NOTE: in RobustSAM, the values of var_limit are invalid (should be in [0, 255]). I think the authors don't notice it.
    # According to the implemenation for backward compatibility in GaussNoise, it will fallback to around (0.2, 0.44).
    # So, I use the default values here instead.
    transforms = A.Compose(
        [A.GaussNoise(std_range=(0.2, 0.44), per_channel=True, p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_iso_noise(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.ISONoise(color_shift=(0.4, 0.5), intensity=(0.7, 0.8), p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_multiplicative_noise(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.MultiplicativeNoise(multiplier=(1.4, 1.5), p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


# TODO: do we really need this? For example, in the DETR, the augmentations already have random resize to smaller and resize to larger.
def apply_resampling_blur(image: np.ndarray, **kwargs):
    resize_factor = 4
    ori_height, ori_width = image.shape[0], image.shape[1]
    new_height, new_width = (
        int(image.shape[0] / resize_factor),
        int(image.shape[1] / resize_factor),
    )
    img_down = cv2.resize(image, (new_width, new_height))
    image = cv2.resize(img_down, (ori_width, ori_height))
    return image


def apply_motion_blur(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.MotionBlur(blur_limit=35, p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_zoom_blur(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.ZoomBlur(max_factor=1.33, step_factor=(0.05, 0.07), p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_color_jitter(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.ColorJitter(brightness=0.7, contrast=0.3, saturation=0.7, hue=0.7, p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_compression(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.ImageCompression(quality_range=(5, 10), p=1)],
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_elastic_transform(image: np.ndarray, **kwargs):
    # NOTE: in the testing stage, due to the limitation of COCO evaluation, they usually load from json file to evaluate the predictions in image scale.
    # But in RobustSAM, they ignore the changes in annotations (mask). I think the changes are minor.
    # Therefore, I still include it as one of degradations.
    # NOTE: the affine transformation is removed from ElasticTransform.
    transforms = A.Compose(
        [A.ElasticTransform(alpha=100, sigma=10, interpolation=cv2.INTER_LANCZOS4, p=1)]
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_glass_blur(image: np.ndarray, **kwargs):
    transforms = A.Compose([A.GlassBlur(max_delta=3, mode="exact", p=1)])
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_brightness(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, -0.2),
                contrast_limit=0,
                brightness_by_max=False,
                p=1,
            )
        ]
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_contrast(image: np.ndarray, **kwargs):
    transforms = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.6, 0.7), p=1)]
    )
    image = _apply_albu_transforms(transforms, image=image)["image"]
    return image


def apply_degradation(
    image: np.ndarray,
    identity: bool = True,
    ignore_transforms: Union[list[str], set[str]] = [],
    **kwargs,
):
    transforms = set(DEGRADATION_TRANSFORMS.keys())
    ignore_transforms = set(ignore_transforms)
    if not identity:
        ignore_transforms.add("identity")

    assert len(transforms & ignore_transforms) == len(ignore_transforms)
    transforms -= ignore_transforms

    name = random.choice(list(transforms))
    image = DEGRADATION_TRANSFORMS[name](image, **kwargs)
    return image


DEGRADATION_TRANSFORMS = dict(
    snow=apply_snow,
    fog=apply_fog,
    rain=apply_rain,
    gaussian_noise=apply_gaussian_noise,
    iso_noise=apply_iso_noise,
    multiplicative_noise=apply_multiplicative_noise,
    resampling_blur=apply_resampling_blur,
    motion_blur=apply_motion_blur,
    zoom_blur=apply_zoom_blur,
    color_jitter=apply_color_jitter,
    compression=apply_compression,
    elastic=apply_elastic_transform,
    glass_blur=apply_glass_blur,
    brightness=apply_brightness,
    contrast=apply_contrast,
    identity=lambda image, **kwargs: image,
)
