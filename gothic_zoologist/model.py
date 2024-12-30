from flax import linen as nn

# data:           images 256x144 - 3 channels (rgb)
# after 1 conv:   images 256x144 - 32 channels,
# after avg pool: images 128x72  - 32 channels,
# after 2 conv:   images 128x72  - 64 channels,
# after avg pool: images 64x36   - 64 channels,
# reshape -> matrix 1x147456 -> (147456 = 36 * 64 * 64)
# dense -> layer of 256 neurons (hidden layer)
# dense -> output layer
# log_softmax -> percentage probability

class CNN(nn.Module):
    outputs: int

    @nn.compact
    def __call__(self, x):
        # input x - (144x256x3)

        # basic convolution (144x256x32)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (72x128x32)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # details convolution (72x128x64)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (36x64x64)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # convert 2d image to a flat single dimension vector (36x64x64) -> (36 * 64 * 64) -> (1x147456)
        x = x.reshape((-1,))

        # hidden layer
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # outputs
        x = nn.Dense(features=self.outputs)(x)
        x = nn.softmax(x, axis=-1)

        return x