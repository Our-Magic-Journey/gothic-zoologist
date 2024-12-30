from flax import linen as nn

# data:           images 128x128 - 3 channels (rgb)
# after 1 conv:   images 128x128 - 32 channels,
# after avg pool: images 64x64   - 32 channels,
# after 2 conv:   images 64x64   - 64 channels,
# after avg pool: images 32x32   - 64 channels,
# reshape -> matrix 1x65536 -> (65536 = 32 * 32 * 64)
# dense -> layer of 256 neurons (hidden layer)
# dense -> output layer
# log_softmax -> percentage probability

class CNN(nn.Module):
    outputs: int

    @nn.compact
    def __call__(self, x):
        # input x - (128x128x3)

        # basic convolution (128x128x32)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (64x64x32)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # details convolution (64x64x64)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (32x32x64)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # convert 2d image to a flat single dimension vector (36x64x64) -> (32 * 32 * 64) -> (1x65536)
        x = x.reshape((-1,))

        # hidden layer
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # outputs
        x = nn.Dense(features=self.outputs)(x)
        x = nn.log_softmax(x, axis=-1)

        return x