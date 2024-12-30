import jax
import optax
import os

import dataset
from model import CNN
from jax import random
from jax import numpy as jnp
from flax.training import train_state
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import time

def train_model():
    rng, inp_rng, init_rng = random.split(random.PRNGKey(42), 3)

    print("loading dataset")
    (test, train, categories) = dataset.load_gothic_dataset()

    print("loading model")
    model = CNN(outputs=len(categories))
    params = model.init(rng, random.normal(inp_rng, (128, 128, 3)))
    optimizer = optax.adam(learning_rate=1e-4)
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    num_epochs = 50

    print(f"starting training on {len(train)} images")
    for epoch in range(num_epochs):
        start_time = time.time()
        model_state, train_loss, train_accuracy = train_epoch(model_state, train)
        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Train Accuracy: {train_accuracy:.8f}, Time: {epoch_time:.2f}s')

        if train_loss < 0.001:
            break

    print("training finished")
    create_checkpoint(categories, model_state.params)

    return model_state


def load_checkpoint() -> (int, jnp.ndarray) or None:
    if not os.path.exists("/app/.checkpoints/single_save"):
        return None

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer. restore("/app/.checkpoints/single_save")

    return raw_restored['config']['categories'], raw_restored['params']


def create_checkpoint(categories, params):
    print("saving checkpoint")

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    config = {'categories': categories}
    ckpt = {'params': params, 'config': config }
    save_args = orbax_utils.save_args_from_target(ckpt)

    ocp.test_utils.erase_and_create_empty('/app/.checkpoints/')
    orbax_checkpointer.save('/app/.checkpoints/single_save', ckpt, save_args=save_args)
    print("checkpoint saved")


@jax.jit
def apply_model(state, image, label_id):
    def loss_fn(params):
        prediction = state.apply_fn(params, image)
        one_hot = jax.nn.one_hot(label_id, prediction.shape[0])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=prediction, labels=one_hot))

        return loss, prediction


    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, prediction), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(prediction, -1) == label_id)

    return grads, loss, accuracy


def train_epoch(state, images):
    epoch_loss = []
    epoch_accuracy = []

    for (image, label) in images:
        grads, loss, accuracy = apply_model(state, image, label)
        state = state.apply_gradients(grads=grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = jnp.mean(jnp.array(epoch_loss))
    train_accuracy = jnp.mean(jnp.array(epoch_accuracy))

    return state, train_loss, train_accuracy


def verify_training(model: CNN, params, test_data):
    test_loss = []
    test_accuracy = []

    @jax.jit
    def apply_model(params, image, label_id):
        def loss_fn(params):
            prediction = model.apply(params, image)
            one_hot = jax.nn.one_hot(label_id, prediction.shape[0])
            loss = jnp.mean(optax.softmax_cross_entropy(logits=prediction, labels=one_hot))

            return loss, prediction


        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, prediction), grads = grad_fn(params)
        accuracy = jnp.mean(jnp.argmax(prediction, -1) == label_id)

        return grads, loss, accuracy


    for (img, label) in test_data:
        _, loss, accuracy = apply_model(params, img, label)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    print(f'loss: {jnp.mean(jnp.array(test_loss)):.4f}, accuracy: {jnp.mean(jnp.array(test_accuracy)):.4f}')