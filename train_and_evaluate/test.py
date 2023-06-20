import torch
from torch.utils.data import DataLoader
from train_and_evaluate.trainer import Trainer
from train_and_evaluate.dataset import MeasurementDataset
from nn_no_preaugmentation.mlp import Mlp, get_network_prediction


def get_model_predictions_on_test_dataset(restored_checkpoint, checkpoint_folder, output_classes, input_features,
                                          test_x_directory, test_y_directory,
                                          batch_size, num_prev_steps, train_x_directory, train_y_directory, normalize=True):
    train_dataset = MeasurementDataset(x_directory=train_x_directory,
                                       y_directory=train_y_directory,
                                       num_features=input_features,
                                       num_prev_steps=num_prev_steps,
                                       augmentation_count=0,
                                       augmentation_distance_m=0,
                                       is_training=True,
                                       normalize=normalize
                                       )
    train_min_max = train_dataset.get_training_min_max()

    test_data = MeasurementDataset(x_directory=test_x_directory,
                                   y_directory=test_y_directory,
                                   num_features=input_features,
                                   num_prev_steps=num_prev_steps,
                                   augmentation_count=0,
                                   augmentation_distance_m=0,
                                   is_training=False,
                                   training_set_min_max=train_min_max,
                                   normalize=normalize
                                   )

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = Mlp(input_features=input_features * num_prev_steps, output_classes=output_classes)
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, crit, optim=optim, checkpoint_folder=checkpoint_folder)
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


if __name__ == "__main__":
    # dataset parameters
    GRID_WIDTH = 800
    GRID_HEIGHT = 800
    grid_element_length = 50
    num_prev_steps = 3
    input_features = 6
    restored_checkpoint = 250
    normalize = True
    # dataset parameters

    output_classes = int((GRID_WIDTH / grid_element_length) * (GRID_HEIGHT / grid_element_length))
    network_input_length = num_prev_steps * input_features

    checkpoint_folder = f"checkpoints/mlp_{input_features}_grid{grid_element_length}_prev{num_prev_steps}{'_normalized' if normalize else ''}_minadjusted"

    train_x_directory = f"../datasets/erlangen_dataset_minadjusted_gridlen{grid_element_length}.csv"
    train_y_directory = f"../datasets/erlangen_dataset_minadjusted_gridlen{grid_element_length}_label.csv"

    test_x_directory = f"../datasets/erlangen_test_dataset_minadjusted_gridlen{grid_element_length}.csv"
    test_y_directory = f"../datasets/erlangen_test_dataset_minadjusted_gridlen{grid_element_length}_label.csv"

    get_model_predictions_on_test_dataset(restored_checkpoint=restored_checkpoint,
                                          checkpoint_folder=checkpoint_folder,
                                          output_classes=output_classes,
                                          input_features=input_features,
                                          test_x_directory=test_x_directory,
                                          test_y_directory=test_y_directory,
                                          batch_size=32,
                                          num_prev_steps=num_prev_steps,
                                          train_x_directory=train_x_directory,
                                          train_y_directory=train_y_directory,
                                          normalize=normalize
                                          )
