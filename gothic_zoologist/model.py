from flax import linen as nn

# data:           images 32x32 - 3 channels (rgb), batch: 128 images
# after 1 conv:   images 32x32 - 32 channels,      batch: 128 images
# after avg pool: images 16x16 - 32 channels,      batch: 128 images
# after 2 conv:   images 16x16 - 64 channels,      batch: 128 images
# after avg pool: images 08x08 - 64 channels,      batch: 128 images
# reshape -> matrix 128x4096 -> (4096 = 8 * 8 * 64)
# dense -> layer of 256 neurons (hidden layer)
# dense -> output layer
# log_softmax -> percentage probability

class CNN(nn.Module):
    outputs: int

    @nn.compact
    def __call__(self, x):
        # input x - (128x32x32x3)

        # basic convolution (128x32x32x32)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (128x16x16x32)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # details convolution (128x16x16x64)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        # reduce size (128x8x8x64)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # convert 2d image to a flat single dimension vector (128x8x8x64) -> (128x8*8*64) -> (128x4096)
        x = x.reshape((x.shape[0], -1))

        # hidden layer (128x256)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # outputs (128xOutputes)
        x = nn.Dense(features=self.outputs)(x)
        x = nn.log_softmax(x, axis=-1)

        return x