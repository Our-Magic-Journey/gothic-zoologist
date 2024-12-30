from flax import linen as nn

# data:           images 256x144 - 3 channels (rgb), batch: 20 images
# after 1 conv:   images 256x144 - 32 channels,      batch: 20 images
# after avg pool: images 128x72  - 32 channels,      batch: 20 images
# after 2 conv:   images 128x72  - 64 channels,      batch: 20 images
# after avg pool: images 64x36   - 64 channels,      batch: 20 images
# reshape -> matrix 20x147456 -> (147456 = 64 * 36 * 64)
# dense -> layer of 256 neurons (hidden layer)
# dense -> output layer
# log_softmax -> percentage probability

class CNN(nn.Module):
    outputs: int

    @nn.compact
    def __call__(self, x):
        # input x - (20x256x144x3)

        # basic convolution (20x256x144x32)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (20x128x72x32)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # details convolution (20x128x72x64)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (20x64x36x64)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # convert 2d image to a flat single dimension vector (20x64x36x64) -> (20x 64 * 36 * 64) -> (20x147456)
        x = x.reshape((x.shape[0], -1))

        # hidden layer (20x256)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # outputs (20xOutputes)
        x = nn.Dense(features=self.outputs)(x)
        x = nn.log_softmax(x, axis=-1)

        return x