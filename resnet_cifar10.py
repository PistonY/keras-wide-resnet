from keras.initializers import he_normal
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D, ZeroPadding3D
from keras.layers.core import Activation, Dense, Reshape
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2


def lr_schedule(epoch):
    if epoch < 60:
        return 0.1
    elif epoch < 120:
        return 0.1 * 0.2
    elif epoch < 160:
        return 0.1 * (0.2 ** 2)
    else:
        return 0.1 * (0.2 ** 3)


def residual_projection_block(x0, filters, first_unit=False, down_sample=False):
    residual_kwargs = {
        'kernel_size': (3, 3),
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': he_normal(),
        'kernel_regularizer': l2(5e-4)
    }
    skip_kwargs = {
        'kernel_size': (1, 1),
        'padding': 'valid',
        'use_bias': False,
        'kernel_initializer': he_normal(),
        'kernel_regularizer': l2(5e-4)
    }

    if first_unit:
        x0 = BatchNormalization(momentum=.9)(x0)
        x0 = Activation('relu')(x0)
        if down_sample:
            residual_kwargs['strides'] = (2, 2)
            skip_kwargs['strides'] = (2, 2)
        x1 = Conv2D(filters, **residual_kwargs)(x0)
        x1 = BatchNormalization(momentum=.9)(x1)
        x1 = Activation('relu')(x1)
        residual_kwargs['strides'] = (1, 1)
        x1 = Conv2D(filters, **residual_kwargs)(x1)
        x0_shape = x0.shape.as_list()[1:]
        x1_shape = x1.shape.as_list()[1:]
        if x0_shape != x1_shape:
            x0 = Conv2D(filters, **skip_kwargs)(x0)
    else:
        x1 = BatchNormalization(momentum=.9)(x0)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(filters, **residual_kwargs)(x1)
        x1 = BatchNormalization(momentum=.9)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(filters, **residual_kwargs)(x1)
    x0 = Add()([x0, x1])
    return x0


def residual_zero_padding_block(x0, filters, first_unit=False,
                                down_sample=False):
    residual_kwargs = {
        'kernel_size': (3, 3),
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': he_normal(),
        'kernel_regularizer': l2(5e-4)
    }
    skip_kwargs = {
        'kernel_size': (1, 1),
        'padding': 'valid',
        'use_bias': False,
        'kernel_initializer': he_normal(),
        'kernel_regularizer': l2(5e-4)
    }

    if first_unit:
        x0 = BatchNormalization(momentum=.9)(x0)
        x0 = Activation('relu')(x0)
        if down_sample:
            residual_kwargs['strides'] = (2, 2)
            skip_kwargs['strides'] = (2, 2)
        x1 = Conv2D(filters, **residual_kwargs)(x0)
        x1 = BatchNormalization(momentum=.9)(x1)
        x1 = Activation('relu')(x1)
        residual_kwargs['strides'] = (1, 1)
        x1 = Conv2D(filters, **residual_kwargs)(x1)
        x0_img_shape = x0.shape.as_list()[1:-1]
        x1_img_shape = x1.shape.as_list()[1:-1]
        x0_filters = x0.shape.as_list()[-1]
        x1_filters = x1.shape.as_list()[-1]
        if x0_img_shape != x1_img_shape:
            x0 = Conv2D(x0_filters, **skip_kwargs)(x0)
        if x0_filters != x1_filters:
            target_shape = (x1_img_shape[0], x1_img_shape[1], x0_filters, 1)
            x0 = Reshape(target_shape)(x0)
            zero_padding_size = x1_filters - x0_filters
            x0 = ZeroPadding3D(((0, 0), (0, 0), (0, zero_padding_size)))(x0)
            target_shape = (x1_img_shape[0], x1_img_shape[1], x1_filters)
            x0 = Reshape(target_shape)(x0)
    else:
        x1 = BatchNormalization(momentum=.9)(x0)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(filters, **residual_kwargs)(x1)
        x1 = BatchNormalization(momentum=.9)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(filters, **residual_kwargs)(x1)
    x0 = Add()([x0, x1])
    return x0


def residual_blocks(x0, filters, n, down_sample=False):
    for i in range(n):
        if i == 0:
            x0 = residual_zero_padding_block(x0, filters, True, down_sample)
        else:
            x0 = residual_zero_padding_block(x0, filters)
    return x0


def residual_network(n=4):
    kernel_kwargs = {
        'kernel_initializer': he_normal(),
        'kernel_regularizer': l2(5e-4)
    }

    inputs = Input((32, 32, 3))

    x0 = Conv2D(16, (3, 3), padding='same', **kernel_kwargs)(inputs)

    x0 = residual_blocks(x0, 16, n)

    x0 = residual_blocks(x0, 32, n, True)

    x0 = residual_blocks(x0, 64, n, True)

    x0 = BatchNormalization(momentum=.9)(x0)
    x0 = Activation('relu')(x0)

    x0 = GlobalAveragePooling2D()(x0)

    x0 = Dense(10, **kernel_kwargs)(x0)
    outputs = Activation('softmax')(x0)

    return Model(inputs=inputs, outputs=outputs)


def wide_residual_blocks(x0, filters, n, k, down_sample=False):
    for i in range(n):
        if i == 0:
            x0 = residual_zero_padding_block(x0, filters * k, True, down_sample)
        else:
            x0 = residual_zero_padding_block(x0, filters * k)
    return x0


def wide_residual_network(n=2, k=8):
    kernel_kwargs = {
        'kernel_initializer': he_normal(),
        'kernel_regularizer': l2(5e-4)
    }

    inputs = Input((32, 32, 3))

    x0 = Conv2D(16, (3, 3), padding='same', **kernel_kwargs)(inputs)

    x0 = wide_residual_blocks(x0, 16, n, k)

    x0 = wide_residual_blocks(x0, 32, n, k, True)

    x0 = wide_residual_blocks(x0, 64, n, k, True)

    x0 = BatchNormalization(momentum=.9)(x0)
    x0 = Activation('relu')(x0)

    x0 = GlobalAveragePooling2D()(x0)

    x0 = Dense(10, **kernel_kwargs)(x0)
    outputs = Activation('softmax')(x0)

    return Model(inputs=inputs, outputs=outputs)


def main():
    import os
    import json

    import numpy as np
    from keras.callbacks import LearningRateScheduler
    from keras.datasets import cifar10
    from keras.losses import categorical_crossentropy
    from keras.optimizers import SGD
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    (cifar10_x0, cifar10_y0), (cifar10_x1, cifar10_y1) = cifar10.load_data()
    cifar10_x = np.concatenate((cifar10_x0, cifar10_x1))
    cifar10_y = np.concatenate((cifar10_y0, cifar10_y1))

    cifar10_x = cifar10_x.astype('float32') / 255
    cifar10_y = to_categorical(cifar10_y, 10)

    train_x, valid_x, train_y, valid_y = train_test_split(
        cifar10_x, cifar10_y, test_size=10000
    )

    batch_size = 128
    epochs = 200

    model = wide_residual_network(n=4, k=10)
    optimizer = SGD(momentum=.9, nesterov=True)
    model.compile(
        loss=categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    datagen = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True
    )

    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit_generator(
        datagen.flow(train_x, train_y, batch_size=batch_size),
        steps_per_epoch=train_x.shape[0] // batch_size,
        epochs=epochs,
        callbacks=[lr_scheduler],
        verbose=2,
        validation_data=(valid_x, valid_y)
    )

    path = f'var/log/resnet_cifar10_history.json'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w+', encoding='UTF-8') as f:
        json.dump(history.history, f)


if __name__ == '__main__':
    main()
