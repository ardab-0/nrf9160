import torch
from torch.utils.data import DataLoader
from trainer import Trainer
from dataset import MeasurementDataset
from mlp import Mlp, get_network_prediction
import pandas as pd


restored_checkpoint = 200

batch_size = 128
checkpoint_folder = "checkpoints/mlp_12_grid200"
output_classes = 16
input_features = 12

train_x_directory="../saved_measurements/erlangen_dataset_200_augmented.csv"
test_x_directory = "../saved_measurements/erlangen_test_dataset_200_augmented.csv"
test_y_directory="../saved_measurements/erlangen_test_dataset_200_label.csv"


data_x_df = pd.read_csv(train_x_directory)
x_mean = data_x_df.mean()
x_std = data_x_df.std()

test_data = MeasurementDataset(x_directory=test_x_directory,
                                  y_directory=test_y_directory, x_normalization=(x_mean, x_std), num_features=input_features)
test_dataloader = DataLoader(test_data, batch_size=batch_size)




model =  Mlp(input_features=input_features, output_classes=output_classes)
crit = torch.nn.CrossEntropyLoss()
trainer = Trainer(model, crit, checkpoint_folder=checkpoint_folder)
trainer.restore_checkpoint(restored_checkpoint)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


average_accuracy = 0

for input_batch, label_batch in test_dataloader:

    input_batch = input_batch.to(device)
    label_batch = label_batch.to(device)

    # make a prediction
    model_output = trainer.test_model(input_batch)
    predictions = get_network_prediction(model_output)
    equal_count = torch.sum((predictions[0]==label_batch)).item()
    accuracy = equal_count / len(label_batch)
    print("Accuracy in batch: ", accuracy)
    average_accuracy += accuracy

average_accuracy = average_accuracy / len(test_dataloader)
print("Average accuracy: ", average_accuracy )







