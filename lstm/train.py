import pandas as pd
import torch as t
from lstm_dataset import LSTMDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from lstm import LSTMModel

# 50 : 8*7 * 4
# 100: 8*7
# 200: 4*4

restored_checkpoint = -1

batch_size = 128
learning_rate = 1e-3
epochs = 500
checkpoint_folder = "checkpoints/lstm_9_grid5_prev10"
train_ratio = 0.9
output_classes = 64 * 25 * 16
input_features = 9
num_prev_steps = 10

x_directory = "../datasets/erlangen_dataset_gridlen5.csv"
y_directory = "../datasets/erlangen_dataset_gridlen5_label.csv"

x_test_directory = "../datasets/erlangen_test_dataset_gridlen5.csv"
y_test_directory = "../datasets/erlangen_test_dataset_gridlen5_label.csv"

data_x_df = pd.read_csv(x_directory)
# x_mean = data_x_df.mean()
# x_std = data_x_df.std()

full_dataset = LSTMDataset(x_directory=x_directory,
                           y_directory=y_directory,
                           num_features=input_features,
                           num_prev_steps=num_prev_steps
                           )

train_size = int(len(full_dataset) * train_ratio)
train_dataset, val_dataset = t.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

train_dl = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = LSTMModel(input_dim=input_features, hidden_dim=32, layer_dim=1, output_dim=output_classes)

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
figure_name = checkpoint_folder.split("/")[1] + " " + str(restored_checkpoint) + " - " + str(restored_checkpoint + epochs) + ' losses.png'

plt.savefig(figure_name)
plt.show()
