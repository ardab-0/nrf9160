import numpy as np
import torch
from torch.utils.data import DataLoader
from nn.trainer import Trainer
from lstm.lstm_dataset import LSTMDataset
from lstm.lstm import LSTMModel, get_network_prediction
import pandas as pd


def get_model_predictions_on_test_dataset(restored_checkpoint, checkpoint_folder, output_classes, input_features,
                                          x_directory, y_directory, batch_size, num_prev_steps):




    test_data = LSTMDataset(x_directory=x_directory,
                           y_directory=y_directory,
                           num_features=input_features,
                           num_prev_steps=num_prev_steps
                           )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_dim=input_features, hidden_dim=32, layer_dim=1, output_dim=output_classes)
    crit = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, crit, checkpoint_folder=checkpoint_folder)
    trainer.restore_checkpoint(restored_checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    average_accuracy = 0
    all_predictions = []
    all_labels = []

    for i, (input_batch, label_batch) in enumerate(test_dataloader):
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        # make a prediction
        model_output = trainer.test_model(input_batch)
        predictions = get_network_prediction(model_output)

        # print(predictions[0].cpu().detach().numpy().shape)
        all_predictions += predictions[0].cpu().detach().numpy().tolist()
        all_labels += label_batch.cpu().detach().numpy().tolist()

        equal_count = torch.sum((predictions[0] == label_batch)).item()
        accuracy = equal_count / len(label_batch)
        print("Accuracy in batch: ", accuracy)
        average_accuracy += accuracy

    average_accuracy = average_accuracy / len(test_dataloader)
    print("Average accuracy: ", average_accuracy)

    return all_predictions, all_labels


# get_model_predictions_on_test_dataset(restored_checkpoint=200,
#                                       checkpoint_folder="./checkpoints/mlp_12_grid200",
#                                       output_classes=16,
#                                       input_features=12,
#                                       test_x_directory="./datasets/erlangen_test_dataset_gridlen200_augmented.csv",
#                                       test_y_directory="./datasets/erlangen_test_dataset_gridlen200_label.csv",
#                                       batch_size=128)
