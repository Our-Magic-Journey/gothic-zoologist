import os
from jax import numpy as jnp
import PIL.Image as pil


def load_gothic_dataset() -> ([(jnp.array, str)], [(jnp.array, str)]):
    return load_dataset("/app/gothic_zoologist/data/gothic/test"), load_dataset("/app/gothic_zoologist/data/gothic/train")


def load_dataset(base_path) -> [(jnp.array, str)]:
    categories = sorted(os.listdir(base_path))
    images = []

    for category in categories:
        folder = os.path.join(base_path, category)

        for image in  load_images_from_folder(folder):
            images.append((image, category))

    return images


def load_images_from_folder(folder) -> [jnp.array]:
    images = []
    files = sorted(os.listdir(folder))

    for file_name in files:
        if file_name.endswith(".jpg"):
            img_path = os.path.join(folder, file_name)
            images.append(load_image(img_path))

    return images


def load_image(path: str) -> jnp.array:
    img = pil.open(path)
    img.thumbnail((224, 224))

    # convert to np and normalize image
    return jnp.array(img, dtype=jnp.float32) / 255