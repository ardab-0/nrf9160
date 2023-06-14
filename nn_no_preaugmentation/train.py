import pandas as pd
import torch as t
from dataset import MeasurementDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from mlp import Mlp
from utils import generate_combinations


CHECKPOINT_FOLDER = "grid_search_checkpoints"

restored_checkpoint = -1  # -1 for no restoration
epochs = 1000  # train until this epoch
early_stopping_patience = 30



# dataset parameters
GRID_WIDTH = 800
GRID_HEIGHT = 800
grid_element_length = 20

normalize = True

# network parameters
batch_size = 256
learning_rate = 1e-3
train_ratio = 0.9
# num_prev_steps = 3
# input_features = 15
augmentation_count = 8
augmentation_distance_m = 10
# network parameters

num_prev_steps_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
input_features_list = [6, 9, 12, 15, 18]
parameter_lists = [num_prev_steps_list, input_features_list]
parameter_combinations = generate_combinations(parameter_lists)

for param_comb in parameter_combinations:
    num_prev_steps = param_comb[0]
    input_features = param_comb[1]

    output_classes = int((GRID_WIDTH / grid_element_length) * (GRID_HEIGHT / grid_element_length))
    network_input_length = num_prev_steps * input_features

    checkpoint_name = f"{CHECKPOINT_FOLDER}/mlp_{input_features}_grid{grid_element_length}_prev{num_prev_steps}{'_normalized' if normalize else ''}_minadjusted"

    x_directory = f"../datasets/erlangen_dataset_minadjusted_gridlen{grid_element_length}.csv"
    y_directory = f"../datasets/erlangen_dataset_minadjusted_gridlen{grid_element_length}_label.csv"
    grid_line_directory = f"../datasets/erlangen_dataset_minadjusted_grid_lines_gridlen{grid_element_length}.json"

    full_dataset = MeasurementDataset(x_directory=x_directory,
                                      y_directory=y_directory,
                                      num_features=input_features,
                                      num_prev_steps=num_prev_steps,
                                      augmentation_count=augmentation_count,
                                      augmentation_distance_m=augmentation_distance_m,
                                      is_training=True,
                                      normalize=normalize,
                                      grid_line_directory=grid_line_directory
                                      )

    train_size = int(len(full_dataset) * train_ratio)
    generator = t.Generator().manual_seed(
        42)  # to have same split everytime, otherwise during restoration training dataset and validation datasets are mixed
    train_dataset, val_dataset = t.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size],
                                                           generator=generator)

    train_dl = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = Mlp(input_features=network_input_length, output_classes=output_classes)

    crit = t.nn.CrossEntropyLoss()
    optim = t.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, crit=crit, optim=optim, early_stopping_patience=early_stopping_patience, train_dl=train_dl, val_test_dl=val_dl,
                      cuda=True, checkpoint_folder=checkpoint_name)

    trainer.restore_checkpoint(restored_checkpoint)

    res = trainer.fit(epochs=epochs)

# # plot the results
# plt.plot(np.arange(len(res[0])), res[0], label='train loss')
# plt.plot(np.arange(len(res[1])), res[1], label='val loss')
# plt.yscale('log')
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()
