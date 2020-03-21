import pytorch_lightning as pl
from argparse import ArgumentParser
from practical_ai.vision.gan.core import GAN
from practical_ai.vision.gan.hparams import (
    # dataset
    DATASET,
    LATENT_DIM,
    DIM,
    CHANNELS,
    # architecture, loss
    GENERATOR_TYPE,
    DISCRIMINATOR_TYPE,
    ADVERSARIAL_LOSS_TYPE,
    NORM_TYPE,
    DIM_CHANNEL_MULTIPLIER,
    KERNEL_SIZE,
    # training
    BATCH_SIZE,
    LR,
    BETA1,
    BETA2,
) 


def main(args):
    model = GAN(hparams=args)
#     trainer = Trainer.from_argparse_args(args)
    trainer = pl.trainer.Trainer()
    
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser("Train a Generative Adversarial Network")
#     parser = pl.Trainer.add_argparse_args(parser)
    
    # dataset
    parser.add_argument("dataset", type=str, default=DATASET, nargs="?",
                        help="name of dataset")
    parser.add_argument("latent_dim", type=int, default=LATENT_DIM, nargs="?",
                        help="dimensionality of the latent space")
    parser.add_argument("dim", type=int, default=DIM, nargs="?",
                        help="image height / width")
    parser.add_argument("channels", type=int, default=CHANNELS, nargs="?",
                        help="image channels")
    # architecture, loss
    parser.add_argument("generator_type", type=str, default=GENERATOR_TYPE, nargs="?",
                        help="generator type")
    parser.add_argument("discriminator_type", type=str, default=DISCRIMINATOR_TYPE, nargs="?",
                        help="discriminator type")
    parser.add_argument("adversarial_loss_type", type=str, default=ADVERSARIAL_LOSS_TYPE, nargs="?",
                        help="adversarial loss type")
    parser.add_argument("norm_type", type=str, default=NORM_TYPE, nargs="?",
                        help="normalization type")
    parser.add_argument("dim_channel_multiplier", type=int, default=DIM_CHANNEL_MULTIPLIER, nargs="?",
                        help="ratio between dim and channels of the imagees")
    parser.add_argument("kernel_size", type=int, default=KERNEL_SIZE, nargs="?",
                        help="size of kernels which are used in generator and discriminator")
    # training
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, nargs="?",
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=LR, nargs="?",
                        help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=BETA1, nargs="?",
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=BETA2, nargs="?",
                        help="adam: decay of first order momentum of gradient")
    args = parser.parse_args()
    main(args)