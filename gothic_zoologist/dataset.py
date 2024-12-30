import os

import jax
from jax import numpy as jnp
import PIL.Image as pil
import dm_pix

def load_gothic_dataset() -> ([(jnp.array, int)], [(jnp.array, int)], [str]):
    (test_img, test_categories) = load_dataset("/app/gothic_zoologist/data/gothic/test")
    (train_img, train_categories) = load_dataset("/app/gothic_zoologist/data/gothic/train")

    categories = unique(test_categories + train_categories)
    test_img = [(img, categories.index(category)) for (img, category) in test_img]
    train_img = [(img, categories.index(category)) for (img, category) in train_img]

    return test_img, train_img, categories


def load_dataset(base_path) -> ([(jnp.array, str)], [str]):
    categories = sorted(os.listdir(base_path))
    images = []

    for category in categories:
        folder = os.path.join(base_path, category)

        for image in  load_images_from_folder(folder):
            images.append((image, category))

    return images, categories


def load_images_from_folder(folder) -> [jnp.array]:
    images = []
    files = sorted(os.listdir(folder))

    for file_name in files:
        if file_name.endswith(".jpg"):
            img_path = os.path.join(folder, file_name)
            img = load_image(img_path)
            images.append(img)


    return images


def load_image(path: str) -> jnp.array:
    img = pil.open(path)
    img.thumbnail((128, 128))

    # convert to np and normalize image
    normal = jnp.array(img, dtype=jnp.float32) / 255

    return normal


def unique(data: [str]):
    return list(set(data))