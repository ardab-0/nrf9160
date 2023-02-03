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

restored_checkpoint = -1

batch_size = 128
learning_rate = 1e-3
epochs = 200
checkpoint_folder = "checkpoints/mlp_12"
train_ratio = 0.9
output_classes = 56
input_features = 12

x_directory="../saved_measurements/erlangen_dataset_augmented.csv"
y_directory="../saved_measurements/erlangen_dataset_label.csv"


data_x_df = pd.read_csv(x_directory)
x_mean = data_x_df.mean()
x_std = data_x_df.std()

full_dataset = MeasurementDataset(x_directory=x_directory,
                                  y_directory=y_directory, x_normalization=(x_mean, x_std), num_features=input_features)

train_size = int(len(full_dataset) * train_ratio)
train_dataset, val_dataset = t.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

train_dl = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = Mlp(input_features=input_features, output_classes=output_classes)

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
plt.savefig('losses.png')
plt.show()
