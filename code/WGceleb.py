import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display
import os
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')

IMG_SIZE = 64
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

LATENT_DIM = 128
BATCH_SIZE = 128         
EPOCHS = 20
N_CRITIC = 1            
GP_LAMBDA = 10.0         
LEARNING_RATE = 2e-4
BETA_1, BETA_2 = 0.0, 0.9  

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_celeba(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    def _map(ex):
        # tfds 'image' key: H x W x 3, dtype=uint8
        x = tf.image.resize(ex["image"], (img_size, img_size), method="bilinear")
        x = tf.cast(x, tf.float32) / 127.5 - 1.0  # [-1,1]
        return x

    ds, info = tfds.load("celeb_a", split="train", with_info=True)
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    steps_per_epoch = info.splits["train"].num_examples // batch_size
    return ds, steps_per_epoch

train_ds, steps_per_epoch = load_celeba()

def build_generator(latent_dim=LATENT_DIM):
    """
    生成器：输入 z -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
    使用 Conv2DTranspose + BatchNorm + ReLU，最后 Tanh 输出 [-1,1]
    """
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * 1024, use_bias=False)(inputs)
    x = layers.Reshape((4, 4, 1024))(x)

    x = layers.Conv2DTranspose(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(IMG_CHANNELS, 3, padding="same", activation="tanh")(x)
    return keras.Model(inputs, outputs, name="generator")

def build_critic(img_shape=IMG_SHAPE):
    """
    Critic（WGAN-GP）：不使用 Sigmoid，不用 BatchNorm（避免违反 1-Lipschitz 假设）。
    用 LeakyReLU + 少量 Dropout/LayerNorm（可选）。
    """
    def conv_block(x, filters, stride=2, apply_ln=False):
        x = layers.Conv2D(filters, 4, strides=stride, padding="same")(x)
        if apply_ln:
            x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    inputs = keras.Input(shape=img_shape)
    x = conv_block(inputs, 64, 2, apply_ln=False)    # 64x64 -> 32x32
    x = conv_block(x, 128, 2, apply_ln=True)         # 32x32 -> 16x16
    x = conv_block(x, 256, 2, apply_ln=True)         # 16x16 -> 8x8
    x = conv_block(x, 512, 2, apply_ln=True)         # 8x8 -> 4x4
    x = layers.Conv2D(1, 4, strides=1, padding="valid")(x)  # 4x4 -> 1x1
    outputs = layers.Flatten()(x)  # shape (batch, 1)
    return keras.Model(inputs, outputs, name="critic")

generator = build_generator()
critic = build_critic()

class WGAN_GP(keras.Model):
    def __init__(self, generator, critic, latent_dim, GP_lambda=10.0, n_critic=5):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.GP_lambda = GP_lambda
        self.n_critic = n_critic

        self.gen_steps = keras.metrics.Mean(name="g_loss")
        self.cri_steps = keras.metrics.Mean(name="c_loss")

    def compile(self, g_optimizer, c_optimizer):
        super().compile(run_eagerly=False)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

    def gradient_penalty(self, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * real_images + (1 - epsilon) * fake_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(x_hat)
            pred = self.critic(x_hat, training=True)
        grads = gp_tape.gradient(pred, x_hat)

        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]) + 1e-12)
        gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]

        c_losses = []
        for _ in range(self.n_critic):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape_c:
                fake_images = self.generator(random_latent_vectors, training=True)
                real_logits = self.critic(real_images, training=True)
                fake_logits = self.critic(fake_images, training=True)

                c_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(real_images, fake_images)
                c_loss_total = c_loss + self.GP_lambda * gp

            grads_c = tape_c.gradient(c_loss_total, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables))
            c_losses.append(c_loss_total)


        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape_g:
            generated_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.critic(generated_images, training=True)

            g_loss = -tf.reduce_mean(fake_logits)

        grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        self.cri_steps.update_state(tf.reduce_mean(c_losses))
        self.gen_steps.update_state(g_loss)
        return {"c_loss": self.cri_steps.result(), "g_loss": self.gen_steps.result()}

    @property
    def metrics(self):
        return [self.cri_steps, self.gen_steps]

class WGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=LATENT_DIM, grid=(4,4), outdir="wgan_gp_samples", seed=SEED):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.grid = grid
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        rng = np.random.default_rng(seed)
        self.fixed_noise = rng.standard_normal(size=(num_img, latent_dim)).astype("float32")

    def on_epoch_end(self, epoch, logs=None):
        imgs = self.model.generator(self.fixed_noise, training=False)
        imgs = (imgs * 0.5 + 0.5).numpy()  # [-1,1] -> [0,1]

        rows, cols = self.grid
        fig = plt.figure(figsize=(cols*2, rows*2))
        for i in range(rows * cols):
            ax = plt.subplot(rows, cols, i+1)
            ax.imshow(np.clip(imgs[i], 0, 1))
            ax.axis("off")
        fig.suptitle(f"WGAN-GP Samples — Epoch {epoch+1}", y=0.98)
        fig.tight_layout()
        save_path = os.path.join(self.outdir, f"epoch_{epoch+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        display(fig)
        plt.close(fig)

EPOCHS = 20
wgan = WGAN_GP(generator, critic, LATENT_DIM, GP_lambda=GP_LAMBDA, n_critic=N_CRITIC)
wgan.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
    c_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
)

monitor = WGANMonitor(num_img=16, latent_dim=LATENT_DIM, grid=(4,4))
history = wgan.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,   
    callbacks=[monitor],
)
