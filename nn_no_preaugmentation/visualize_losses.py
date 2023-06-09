import matplotlib.pyplot as plt
import torch
import glob


def main():
    # parameters
    checkpoint_name = "checkpoints/mlp_9_grid50_prev3_normalized_minadjusted"
    # parameters

    files = sorted(glob.glob(checkpoint_name + '/*'))
    last_epoch = int((files[-1].split("_")[-1]).split(".")[0])
    ckp = torch.load('{}/checkpoint_{:03d}.ckp'.format(checkpoint_name, last_epoch))

    train_losses = ckp['train_loss']
    validation_losses = ckp['validation_loss']

    # plot the results
    plt.plot(train_losses, label='train loss')
    plt.plot(validation_losses, label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(checkpoint_folder.split("/")[1])
    plt.show()


if __name__ == "__main__":
    main()
