import matplotlib.pyplot as plt
import torch
import glob


def main():
    # parameters
    checkpoint_folder = "checkpoints/mlp_21_grid50_prev15_normalized"
    # parameters

    files = sorted(glob.glob(checkpoint_folder + '/*'))
    last_epoch = int((files[-1].split("_")[-1]).split(".")[0])
    ckp = torch.load('{}/checkpoint_{:03d}.ckp'.format(checkpoint_folder, last_epoch))

    train_losses = ckp['train_loss']
    validation_losses = ckp['validation_loss']

    # plot the results
    plt.plot(train_losses, label='train loss')
    plt.plot(validation_losses, label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
