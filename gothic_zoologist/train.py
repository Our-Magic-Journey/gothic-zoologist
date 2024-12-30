import jax
import optax
import dataset
from model import CNN
from jax import random
from jax import numpy as jnp
from flax.training import train_state

def train_model():
    rng, inp_rng, init_rng = random.split(random.PRNGKey(42), 3)

    print("loading dataset")
    (test, train, categories) = dataset.load_gothic_dataset()

    print("loading model")
    model = CNN(outputs=len(categories))
    params = model.init(rng, random.normal(inp_rng, (144, 256, 3)))
    optimizer = optax.adam(learning_rate=1e-4)
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    num_epochs = 3

    print("starting training")
    for epoch in range(num_epochs):
        model_state, train_loss, train_accuracy = train_epoch(model_state, train)
        print(f'epoch: {epoch:03d}, train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}')

    print("training finished")
    verify_training(model_state, test)

    return model_state


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


def verify_training(trained_model_state, test_data):
    test_loss = []
    test_accuracy = []


    for (img, label) in test_data:
        _, loss, accuracy = apply_model(trained_model_state, img, label)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    print(f'loss: {jnp.mean(jnp.array(test_loss)):.4f}, accuracy: {jnp.mean(jnp.array(test_accuracy)):.4f}')