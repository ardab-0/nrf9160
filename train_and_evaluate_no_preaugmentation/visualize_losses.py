import matplotlib.pyplot as plt
import torch
import glob


def main():
    # parameters
    CHECKPOINT_FOLDER = "grid_search_checkpoints"
    GRID_WIDTH = 800
    GRID_HEIGHT = 800
    grid_element_length = 50
    normalize = True
    augmentation_count = 0
    augmentation_distance_m = 3
    # num_prev_steps_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # input_features_list = [6, 9, 12, 15, 18]
    input_features = 6
    num_prev_steps = 10


    checkpoint_name = f"{CHECKPOINT_FOLDER}/mlp_{input_features}_grid{grid_element_length}_prev{num_prev_steps}{'_normalized' if normalize else ''}_minadjusted{'_augmented' + str(augmentation_count) + '-' + str(augmentation_distance_m) if augmentation_count > 0 else ''}"

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
    plt.title(checkpoint_name.split("/")[1])
    plt.show()


if __name__ == "__main__":
    main()
