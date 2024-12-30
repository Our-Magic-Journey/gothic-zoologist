from model import CNN
from jax import random
import train
import orbax
import orbax.checkpoint as ocp
import jax
import dataset

def main():
    ckpt_dir = "/app/.checkpoints/"

    print("loading checkpoint")
    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)

    checkpoint = train.load_checkpoint()

    if checkpoint is None:
        print("checkpoint not found")
        train.train_model()

    print("loading checkpoint")
    outputs, params = train.load_checkpoint()
    model = CNN(outputs=outputs)

    (test_img, _, _) = dataset.load_gothic_dataset()
    print(model.apply(params, test_img[0][0]))


if __name__ == '__main__':
    main()
