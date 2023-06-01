import numpy as np
import torch
from torch.utils.data import DataLoader
from nn.trainer import Trainer
from nn.dataset import MeasurementDataset
from nn.mlp import Mlp, get_network_prediction
import pandas as pd


def get_model_predictions_on_test_dataset(restored_checkpoint, checkpoint_folder, output_classes, input_features,
                                          test_x_directory, test_y_directory,
                                          batch_size, num_prev_steps):


    test_data = MeasurementDataset(x_directory=test_x_directory,
                                   y_directory=test_y_directory,
                                   num_features=input_features,
                                   num_prev_steps=num_prev_steps,
                                   )

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = Mlp(input_features=input_features * num_prev_steps, output_classes=output_classes)
    crit = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, crit, checkpoint_folder=checkpoint_folder)
    trainer.restore_checkpoint(restored_checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    average_accuracy = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for i, (input_batch, label_batch) in enumerate(test_dataloader):
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        # make a prediction
        model_output = trainer.test_model(input_batch)
        predicted_labels, probability = get_network_prediction(model_output)

        # print(predictions[0].cpu().detach().numpy().shape)
        all_predictions += predicted_labels.cpu().detach().numpy().tolist()
        all_labels += label_batch.cpu().detach().numpy().tolist()
        all_probabilities += probability.cpu().detach().numpy().tolist()

        equal_count = torch.sum((predicted_labels == label_batch)).item()
        accuracy = equal_count / len(label_batch)
        print("Accuracy in batch: ", accuracy)
        average_accuracy += accuracy

    average_accuracy = average_accuracy / len(test_dataloader)
    print("Average accuracy: ", average_accuracy)

    return all_predictions, all_labels, all_probabilities



# prediction_grid_indices, label_grid_indices = get_model_predictions_on_test_dataset(
#             restored_checkpoint=29,
#             checkpoint_folder="./checkpoints/mlp_9_grid50_prev5",
#             output_classes=64 * 4,
#             input_features=9,
#             test_x_directory="../datasets/erlangen_test_dataset_gridlen50.csv",
#             test_y_directory="../datasets/erlangen_test_dataset_gridlen50_label.csv",
#             batch_size=32,
#             num_prev_steps=5)

if __name__ == "__main__":
    get_model_predictions_on_test_dataset(restored_checkpoint=12,
                                          checkpoint_folder="./checkpoints/mlp_9_grid10_prev1",
                                          output_classes=64*25*4,
                                          input_features=9,
                                          test_x_directory="../datasets/erlangen_test_dataset_gridlen10.csv",
                                          test_y_directory="../datasets/erlangen_test_dataset_gridlen10_label.csv",
                                          batch_size=32,
                                          num_prev_steps=1)
