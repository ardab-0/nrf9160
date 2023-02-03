import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, input_features, output_classes):
        super(Mlp, self).__init__()

        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = self.softmax(x)
        return x


def get_network_prediction(network_output):

    out = F.softmax(network_output, dim=1)
    predicted_labels = torch.argmax(out, dim=1)
    lane_probability = out[:, 1]

    return predicted_labels, lane_probability
