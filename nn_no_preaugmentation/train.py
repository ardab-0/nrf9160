import pandas as pd
import torch as t
from dataset import MeasurementDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from mlp import Mlp

# 50 : 8*7 * 4
# 100: 8*7
# 200: 4*4

restored_checkpoint = -1 # -1 for no restoration

batch_size = 64
learning_rate = 1e-3
epochs = 300  # train until this epoch
checkpoint_folder = "checkpoints/mlp_18_grid50_prev15_normalized"
train_ratio = 0.9

# network parameters
output_classes = 64 * 4
num_prev_steps = 15
input_features = 18
augmentation_count = 0
augmentation_distance_m = 3
network_input_length = num_prev_steps * input_features
# network parameters

x_directory = "../datasets/erlangen_dataset_gridlen50.csv"
y_directory = "../datasets/erlangen_dataset_gridlen50_label.csv"

full_dataset = MeasurementDataset(x_directory=x_directory,
                                  y_directory=y_directory,
                                  num_features=input_features,
                                  num_prev_steps=num_prev_steps,
                                  augmentation_count=augmentation_count,
                                  augmentation_distance_m=augmentation_distance_m,
                                  is_training=True
                                  )

train_size = int(len(full_dataset) * train_ratio)
generator = t.Generator().manual_seed(42) # to have same split everytime, otherwise during restoration training dataset and validation datasets are mixed
train_dataset, val_dataset = t.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size], generator=generator)

train_dl = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = Mlp(input_features=network_input_length, output_classes=output_classes)

crit = t.nn.CrossEntropyLoss()
optim = t.optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer(model, crit=crit, optim=optim, early_stopping_patience=-1, train_dl=train_dl, val_test_dl=val_dl,
                  cuda=True, checkpoint_folder=checkpoint_folder)
if restored_checkpoint > 0:
    trainer.restore_checkpoint(restored_checkpoint)

res = trainer.fit(epochs=epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
