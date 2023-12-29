# adapted from https://github.com/vrkhazaie/tiny-pix2pix/blob/master/tinypix2pix.py
import keras
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import load_model
from keras import backend as K
import numpy as np
import os
import pdb
import sys
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import tensorflow as tf
import argparse

CONST = -35/20 - np.log10(64)

class TinyPix2Pix():
    def __init__(self, model_dir, write_dir, input_shape=(None, None, 1), wgan=False, patchgan_lr=1e-5, 
                 patchgan_wt=10, unet_lr=2e-4, unet_mae=5):
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.write_dir = write_dir
        self.wgan = wgan
        self.patchgan_lr = patchgan_lr
        self.patchgan_wt = patchgan_wt
        self.unet_lr = unet_lr
        self.unet_mae = unet_mae

    def define_unet(self):
        # UNet
        inputs = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = keras.layers.Conv2D(16, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        
        conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        pool2 = keras.layers.Conv2D(32, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        
        conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        pool3 = keras.layers.Conv2D(64, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        
        conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        drop4 = keras.layers.Dropout(0.5)(conv4)
        pool4 = keras.layers.Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)

        conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)

        up6 = keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(drop5)
        merge6 = keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
        merge7 = keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
        merge8 = keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = keras.layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
        merge9 = keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = keras.layers.Conv2D(1, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)

        self.unet = keras.Model(inputs=inputs, outputs=conv9)


    def define_patchgan(self):
        # PatchNet
        inputs = keras.layers.Input(shape=self.input_shape)
        targets = keras.layers.Input(shape=self.input_shape)

        merged = keras.layers.Concatenate()([inputs, targets])

        x = keras.layers.Conv2D(64, 3, strides=2, padding='same')(merged)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(128, 3, padding='same')(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(256, 3, padding='same')(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        self.patchgan = keras.Model(inputs=[inputs, targets], outputs=x)
        # self.patchgan.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy')
        self.patchgan.compile(optimizer=keras.optimizers.RMSprop(lr=self.patchgan_lr), loss=wasserstein_loss)

    def define_tinypix2pix(self):
        self.define_unet()
        self.define_patchgan()
        self.patchgan.trainable = False

        input_source = keras.layers.Input(shape=self.input_shape)
        unet_output = self.unet(input_source)

        patchgan_output = self.patchgan([input_source, unet_output])

        self.tinypix2pix = keras.Model(inputs=input_source, outputs=[patchgan_output, unet_output])
        optimizer = keras.optimizers.Adam(lr=self.unet_lr, beta_1=0.5)
        self.tinypix2pix.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, self.unet_mae])

    def real_samples(self, dataset, write=False):
        x_realA, x_realB, label = next(dataset)
        y_real = np.ones((x_realA.shape[0],) + (int(x_realA.shape[1]//2), int(x_realA.shape[2]//2), 1))
        if write:
            return [x_realA, x_realB], y_real, label
        return [x_realA, x_realB], y_real

    def fake_samples(self, x_real):
        x_fake = self.unet.predict(x_real)
        y_fake = np.zeros((len(x_fake),) + (int(x_real.shape[1]//2), int(x_real.shape[2]//2), 1))
        return x_fake, y_fake

    def gradient_penalty(self, real_images, generated_images):
        alpha = K.random_uniform(shape=[K.shape(real_images)[0], 1, 1, 1], minval=0.0, maxval=1.0)
        differences = generated_images - real_images
        interpolates = real_images + (alpha * differences)

        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            disc_interpolates = self.patchgan([real_images, interpolates])

        gradients = tape.gradient(disc_interpolates, interpolates)
        slopes = K.sqrt(K.sum(K.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = K.mean(K.square(slopes - 1.0))
        
        return gradient_penalty

    def train_discriminator_with_penalty(self, x_realA, x_realB, x_fakeB):
        with tf.GradientTape() as tape:
            fake_images = self.unet(x_realA)
            d_loss_real = K.mean(self.patchgan([x_realA, x_realB]))
            d_loss_fake = K.mean(self.patchgan([x_realA, fake_images]))

            gradient_penalty = self.gradient_penalty(x_realB, fake_images)
            d_loss = d_loss_fake - d_loss_real + self.patchgan_wt * gradient_penalty  # Adjust penalty weight

        gradients_of_discriminator = tape.gradient(d_loss, self.patchgan.trainable_variables)
        self.patchgan.optimizer.apply_gradients(zip(gradients_of_discriminator, self.patchgan.trainable_variables))
        return d_loss

    def fit_init(self, train_generator, disc_epochs, gan_epochs, steps_per_epoch, g_losses, d_losses_1 = None, d_losses_2 = None, d_losses = None):
        os.makedirs(self.model_dir, exist_ok=True)
        self.define_tinypix2pix()

        for epoch in range(gan_epochs):
            print(f"Epoch {epoch + 1}/{disc_epochs}")
            for step in range(steps_per_epoch):
                [x_realA, x_realB], y_real = self.real_samples(train_generator)
                x_fakeB, y_fake = self.fake_samples(x_realA)
                g_loss, _, _ = self.tinypix2pix.train_on_batch(x_realA, [y_real, x_realB])
                g_losses.append(g_loss)

                print(f">Step {step + 1}/{steps_per_epoch}, g[{g_loss:.3f}]")

        for epoch in range(disc_epochs):
            print(f"Epoch {epoch + 1}/{disc_epochs}")
            for step in range(steps_per_epoch):
                [x_realA, x_realB], y_real = self.real_samples(train_generator)
                x_fakeB, y_fake = self.fake_samples(x_realA)
                if self.wgan:
                    d_loss = self.train_discriminator_with_penalty(x_realA, x_realB, x_fakeB)
                    d_losses.append(d_loss)
                    print(f">Step {step + 1}/{steps_per_epoch}, d[{d_loss:.3f}]")
                else:
                    d_loss1 = self.patchgan.train_on_batch([x_realA, x_realB], y_real)
                    d_loss2 = self.patchgan.train_on_batch([x_realA, x_fakeB], y_fake)
                    d_losses_1.append(d_loss1)
                    d_losses_2.append(d_loss2)
                    print(f">Step {step + 1}/{steps_per_epoch}, d1[{d_loss1:.3f}] d2[{d_loss2:.3f}]")

    def fit(self, train_generator, val_generator, epochs, steps_per_epoch, validation_steps, g_losses, 
            d_losses_1=None, d_losses_2=None, d_losses=None, epoch_offset=0, gen_ratio=1):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            for step in range(steps_per_epoch):
                [x_realA, x_realB], y_real = self.real_samples(train_generator)
                x_fakeB, y_fake = self.fake_samples(x_realA)

                if step % gen_ratio == 0:
                    if self.wgan:
                        d_loss = self.train_discriminator_with_penalty(x_realA, x_realB, x_fakeB)
                        d_losses.append(d_loss)
                    else:
                        d_loss1 = self.patchgan.train_on_batch([x_realA, x_realB], y_real)
                        d_loss2 = self.patchgan.train_on_batch([x_realA, x_fakeB], y_fake)
                        d_losses_1.append(d_loss1)
                        d_losses_2.append(d_loss2)
                g_loss, _, _ = self.tinypix2pix.train_on_batch(x_realA, [y_real, x_realB])
                g_losses.append(g_loss)

                if step % gen_ratio == 0:
                    if self.wgan:
                        print(f">Step {step + 1}/{steps_per_epoch}, d[{d_loss:.3f}] g[{g_loss:.3f}]")
                    else:
                        print(f">Step {step + 1}/{steps_per_epoch}, d1[{d_loss1:.3f}] d2[{d_loss2:.3f}] g[{g_loss:.3f}]")
                else:
                    print(f">Step {step + 1}/{steps_per_epoch}, g[{g_loss:.3f}]")

            if val_generator:
                val_loss = self.validate_model(val_generator, validation_steps)
                print(f"Validation Loss at epoch {epoch + 1 + epoch_offset}: {val_loss}")

    def validate_model(self, generator, steps, write=False, extra=''):
        total_val_loss = 0.0
        total_count = 0
        os.makedirs(os.path.join(self.write_dir, extra), exist_ok=True)
        for step in range(steps):
            if write:
                [x_realA, x_realB], y_real, label = self.real_samples(generator, write=write)
            else:
                [x_realA, x_realB], y_real = self.real_samples(generator, write=write)
            x_fakeB, y_fake = self.fake_samples(x_realA)
            if write:
                write_fake = np.uint8((x_fakeB[..., 0] + 1) * 127.5)
                for i in range(write_fake.shape[0]):
                    Image.fromarray(write_fake[i], 'L').save(os.path.join(self.write_dir, extra, label[i] + '.png'))
            total_count += x_fakeB.shape[0]

            if self.wgan:
                d_loss1 = self.patchgan.test_on_batch([x_realA, x_realB], y_real)
                d_loss2 = self.patchgan.test_on_batch([x_realA, x_fakeB], y_fake)
                d_loss = d_loss1 + d_loss2
            else:
                d_loss = self.train_discriminator_with_penalty(x_realA, x_realB, x_fakeB)
            g_loss, _, _ = self.tinypix2pix.test_on_batch(x_realA, [y_real, x_realB])
            total_val_loss += d_loss + g_loss
            total_count += x_fakeB.shape[0]

        return total_val_loss / (2 * steps)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def load_and_preprocess_images(x_folder, y_folder, mode='train', num_samples=-1):
    final_x_folder = os.path.join(x_folder, mode)
    final_y_folder = os.path.join(y_folder, mode)
    x_data = []
    y_data = []
    shapes = []
    names = []
    if num_samples != -1:
        use_list = os.listdir(final_x_folder)[:num_samples]
    else:
        use_list = os.listdir(final_x_folder)

    for file in use_list:
        x_path = os.path.join(final_x_folder, file)
        y_path = os.path.join(final_y_folder, file)
        
        # Load and preprocess x image
        x_img = load_img(x_path, color_mode='grayscale')
        x_arr = img_to_array(x_img) / 127.5 - 1  # Normalize to [-1, 1]
        
        # Load and preprocess y image
        y_img = load_img(y_path, color_mode='grayscale')
        y_arr = img_to_array(y_img) / 127.5 - 1  # Normalize to [-1, 1]
        
        x_data.append(x_arr)
        y_data.append(y_arr)
        shapes.append(x_arr.shape[1])
        names.append(file.split('.')[0])
    return x_data, y_data, shapes, np.array(names)

def generate_batch(train_x, train_y, train_shapes, train_names, batch_size, randomized=False):
    indices = np.arange(len(train_x))
    indices = indices[np.argsort(train_shapes)]
    seed = 1

    while True:
        starts = np.arange(0, len(indices), batch_size)
        if randomized:
            np.random.shuffle(starts)
        for start in starts:
            batch_indices = indices[start : start + batch_size]
            x_batch = [train_x[i] for i in batch_indices]
            y_batch = [train_y[i] for i in batch_indices]
            batch_names = train_names[batch_indices]

            max_width = (max(img.shape[1] for img in x_batch) + 16 - 1)//16 * 16
            padded_x_batch = [np.pad(img, ((0, 0), (0, max_width - img.shape[1]), (0, 0)), mode='constant') for img in x_batch]
            padded_y_batch = [np.pad(img, ((0, 0), (0, max_width - img.shape[1]), (0, 0)), mode='constant') for img in y_batch]

            yield np.array(padded_x_batch), np.array(padded_y_batch), batch_names

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Model Training')

    # Add arguments
    parser.add_argument('--disc_epochs', type=int, default=0, help='Number of epochs for training the discriminator')
    parser.add_argument('--gan_epochs', type=int, default=0, help='Number of epochs for training the GAN')
    parser.add_argument('--epochs', type=int, default=10, help='Total number of epochs for training')
    parser.add_argument('--outer_epochs', type=int, default=3, help='Outer epochs for some specific process')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples, -1 if not debug')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--gen_ratio', type=int, default=1, help='gen to disc train ratio')
    parser.add_argument('--wgan', action='store_true', help='Use WGAN (True/False)')
    parser.add_argument('--eval_only', action='store_true', help='Evaluate model only (True/False)')
    parser.add_argument('--data_path', type=str, default='./data', help='Data dir')
    parser.add_argument('--model_label', type=str, default='test_wgan', help='Model label or name')
    parser.add_argument('--patchgan_lr', type=float, default=1e-5, help='patchgan learning rate')
    parser.add_argument('--patchgan_wt', type=float, default=10, help='patchgan weight')
    parser.add_argument('--unet_lr', type=float, default=2e-4, help='unet learning rate')
    parser.add_argument('--unet_mae', type=float, default=5, help='unet learning rate')
    parser.add_argument('--no_valid', action='store_true', help='no validation (True/False)')

    # Parse arguments
    args = parser.parse_args()
    return args

def main():
    np.random.seed(1)
    args = parse_arguments()
    data_label = args.data_path
    model_label = args.model_label 

    x_folder = f"{data_label}/clean"
    y_folder = f"{data_label}/noise"

    train_x, train_y, train_shapes, train_names = load_and_preprocess_images(x_folder, y_folder, 'train', num_samples=args.num_samples)
    if not args.no_valid:
        val_x, val_y, val_shapes, valid_names = load_and_preprocess_images(x_folder, y_folder, 'valid', num_samples=args.num_samples)
    test_x, test_y, test_shapes, test_names = load_and_preprocess_images(x_folder, y_folder, 'test', num_samples=args.num_samples)

    batch_size = args.batch_size
    train_generator = generate_batch(train_x, train_y, train_shapes, train_names, batch_size=batch_size, randomized=True)
    if not args.no_valid:
        val_generator = generate_batch(val_x, val_y, val_shapes, valid_names, batch_size=batch_size, randomized=False)
    test_generator = generate_batch(test_x, test_y, test_shapes, test_names, batch_size=batch_size, randomized=False)
    model_dir = f"{data_label}/models/{model_label}"
    os.makedirs(model_dir, exist_ok=True)
    write_dir = f"{data_label}/output/{model_label}"
    os.makedirs(write_dir, exist_ok=True)
    model = TinyPix2Pix(input_shape=(None, None, 1), model_dir=model_dir, write_dir=write_dir, wgan=args.wgan,
                        patchgan_wt=args.patchgan_wt, patchgan_lr=args.patchgan_lr, unet_lr=args.unet_lr, unet_mae=args.unet_mae)
    model.define_tinypix2pix()
    print("unet")
    print(model.unet.count_params())
    print("patchgan")
    print(model.patchgan.count_params())

    disc_epochs = args.disc_epochs  
    gan_epochs = args.gan_epochs      
    epochs = args.epochs
    steps_per_epoch = 1 + (len(train_x) - 1) // batch_size
    if not args.no_valid:
        validation_steps = 1 + (len(val_x) - 1) // batch_size
    test_steps = 1 + (len(test_x) - 1) // batch_size

    if args.eval_only:
        model.patchgan.load_weights(os.path.join(model_dir, f'patchgan_epoch_{str(args.outer_epochs)}.h5'))
        model.unet.load_weights(os.path.join(model_dir, f'unet_epoch_{str(args.outer_epochs)}.h5'))
        # model.patchgan = load_model(os.path.join(model_dir, f'patchgan_epoch_{str(args.outer_epochs)}.h5'))
        # model.unet = load_model(os.path.join(model_dir, f'unet_epoch_{str(args.outer_epochs)}.h5'))
        if not args.no_valid:
            model.validate_model(val_generator, validation_steps, write=True, extra=str(args.outer_epochs) + '/valid/')
        test_loss = model.validate_model(test_generator, test_steps, write=True, extra=str(args.outer_epochs) + '/test/')
        print("Test Loss: ", test_loss)
        sys.exit()

    g_losses = []
    if args.wgan:
        d_losses = []
        model.fit_init(train_generator, disc_epochs=disc_epochs, gan_epochs=gan_epochs, steps_per_epoch=steps_per_epoch, 
                g_losses=g_losses, d_losses = d_losses)
    else:
        d_losses_1, d_losses_2 = [], []
        model.fit_init(train_generator, disc_epochs=disc_epochs, gan_epochs=gan_epochs, steps_per_epoch=steps_per_epoch, 
                g_losses = g_losses, d_losses = d_losses_1, d_losses_2 = d_losses_2)
    test_losses = []
    for i in range(1, args.outer_epochs + 1):
        print("Outer Epoch: ", i)
        if args.no_valid:
            val_generator = None
            validation_steps = None
        if args.wgan:
            model.fit(train_generator, val_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                  validation_steps=validation_steps, g_losses=g_losses, d_losses=d_losses, epoch_offset=(i-1)*epochs, gen_ratio=args.gen_ratio)
        else:
            model.fit(train_generator, val_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, 
                      g_losses=g_losses, d_losses_1=d_losses_1, d_losses_2=d_losses_2, epoch_offset=(i-1)*epochs, gen_ratio=args.gen_ratio)
        if not args.no_valid:
            model.validate_model(val_generator, validation_steps, write=True, extra=str(i) + '/valid/')
        test_loss = model.validate_model(test_generator, test_steps, write=True, extra=str(i) + '/test/')
        print("Test Loss: ", test_loss)
        test_losses.append(test_loss)
        model.patchgan.save(f"{model_dir}/patchgan_epoch_{i}.h5")
        model.unet.save(f"{model_dir}/unet_epoch_{i}.h5")

    with open(os.path.join(write_dir, "test.pickle"), 'wb') as handle:
        pickle.dump(test_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.plot(g_losses, label="Generator")
    if args.wgan:
        plt.plot(d_losses, label="Discriminator")
    else:
        plt.plot(d_losses_1, label="Discriminator Real")
        plt.plot(d_losses_2, label="Discriminator Fake")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig(os.path.join(write_dir, "res.png"))


if __name__ == '__main__':
    main()
