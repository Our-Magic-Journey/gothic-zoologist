from model import CNN
from jax import random
import train

def main():
    ckpt_dir = "/app/.checkpoints/"

    model = CNN(outputs=2)
    rng, inp_rng, init_rng = random.split(random.PRNGKey(42), 3)
    params = model.init(rng, random.normal(inp_rng, (144, 256, 3)))

    train_state = train.train_model()


if __name__ == '__main__':
    main()
