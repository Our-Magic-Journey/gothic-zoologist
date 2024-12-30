import jax
import flax
import optax
import dataset

def main():
    print("JAX Version : {}".format(jax.__version__))
    print("Flax Version : {}".format(flax.__version__))
    print("Optax Version : {}".format(optax.__version__))
    print("JAX Backend : {}".format(jax.default_backend()))

    ckpt_dir = "/app/.checkpoints/"
    (test, train) = dataset.load_gothic_dataset()




if __name__ == '__main__':
    main()
