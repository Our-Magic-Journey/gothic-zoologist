import jax
import flax
import optax
import dataset
from model import CNN
from jax import random
from jax import numpy as jnp

def main():
    print("JAX Version : {}".format(jax.__version__))
    print("Flax Version : {}".format(flax.__version__))
    print("Optax Version : {}".format(optax.__version__))
    print("JAX Backend : {}".format(jax.default_backend()))

    ckpt_dir = "/app/.checkpoints/"
    (test, train, categories) = dataset.load_gothic_dataset()

    model = CNN(outputs=len(categories))
    rng, inp_rng, init_rng = random.split(random.PRNGKey(42), 3)

    params = model.init(rng, random.normal(inp_rng, (256, 144, 3)))
    print(jnp.shape(test[0][0]))
    print(len(categories))
    print("Predictions:", model.apply(params, test[0][0]))


if __name__ == '__main__':
    main()
