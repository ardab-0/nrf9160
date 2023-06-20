import numpy as np
import streamlit as st
import pydeck as pdk
import pandas as pd
from geodesic_calculations import point_at, get_distance_and_bearing
import os
import json
import nn.test




# Helper functions

def calculate_grid(tl_lon, tl_lat, tr_lon, tr_lat, bl_lon, bl_lat, grid_length_meter, t_length, l_length,
                   bearing_angle_deg):
    cols = t_length // grid_length_meter
    rows = l_length // grid_length_meter

    horizontal_steps_top = []
    vertical_steps_left = []
    horizontal_steps_bottom = []
    vertical_steps_right = []
    for i in range(cols + 1):
        point_top = point_at((tl_lat, tl_lon), i * grid_length_meter, bearing_angle_deg)
        horizontal_steps_top.append((point_top[1], point_top[0]))

        point_bottom = point_at((bl_lat, bl_lon), i * grid_length_meter, bearing_angle_deg)
        horizontal_steps_bottom.append((point_bottom[1], point_bottom[0]))

    for i in range(rows + 1):
        point_left = point_at((tl_lat, tl_lon), i * grid_length_meter, bearing_angle_deg + 90)
        vertical_steps_left.append((point_left[1], point_left[0]))

        point_right = point_at((tr_lat, tr_lon), i * grid_length_meter, bearing_angle_deg + 90)
        vertical_steps_right.append((point_right[1], point_right[0]))

    return horizontal_steps_top, vertical_steps_left, horizontal_steps_bottom, vertical_steps_right


def calculate_grid_element_center_coordinates(grid_lines, grid_length_meter, bearing_angle_deg):
    grid_element_center_coordinates = []

    N = len(grid_lines["horizontal_lines"])
    M = len(grid_lines["vertical_lines"])

    for i in range(N-1):
        current_point = grid_lines["horizontal_lines"][i]["start"]
        adjusted_point = point_at((current_point[1], current_point[0]), grid_length_meter/2, bearing_angle_deg+90)
        adjusted_point = point_at(adjusted_point, grid_length_meter / 2, bearing_angle_deg)
        points_in_row = []
        for j in range(M-1):
            # lat, lon
            new_point = point_at(adjusted_point, j*grid_length_meter, bearing_angle_deg)
            points_in_row.append((new_point[1], new_point[0]))

        grid_element_center_coordinates.append(points_in_row)

    return grid_element_center_coordinates




def calculate_grid_lines(tl_lon, tl_lat, tr_lon, tr_lat, bl_lon, bl_lat, grid_length, t_length, l_length,
                         bearing_angle_deg):
    horizontal_steps_top, vertical_steps_left, horizontal_steps_bottom, vertical_steps_right = calculate_grid(tl_lon,
                                                                                                              tl_lat,
                                                                                                              tr_lon,
                                                                                                              tr_lat,
                                                                                                              bl_lon,
                                                                                                              bl_lat,
                                                                                                              grid_length,
                                                                                                              t_length,
                                                                                                              l_length,
                                                                                                              bearing_angle_deg)

    vertical_lines = []
    horizontal_lines = []

    for i in range(len(horizontal_steps_top)):
        vertical_lines.append({"start": horizontal_steps_top[i], "end": horizontal_steps_bottom[i]})

    for i in range(len(vertical_steps_left)):
        horizontal_lines.append({"start": vertical_steps_left[i], "end": vertical_steps_right[i]})

    vertical_and_horizontal_lines = {"vertical_lines": vertical_lines,
                                     "horizontal_lines": horizontal_lines}

    return vertical_and_horizontal_lines, len(horizontal_steps_top) - 1, len(
        vertical_steps_left) - 1  # coordinate number is 1 more than the grid element number


def find_grid_index(lon, lat, grid_lines, cols, rows):
    row = -1
    col = -1

    i = 0
    j = 0
    vertical_lines = grid_lines["vertical_lines"]
    horizontal_lines = grid_lines["horizontal_lines"]

    while i < cols:
        if (lon > vertical_lines[i]["start"][0] and lon <= vertical_lines[i + 1]["start"][0]):
            col = i
        i += 1

    while j < rows:
        if (lat < horizontal_lines[j]["start"][1] and lat >= horizontal_lines[j + 1]["start"][1]):
            row = j
        j += 1

    return col, row


def divide_samples(dataset_df, grid_lines):
    sample_positions_on_grid = np.zeros((len(dataset_df), 3), dtype=int)
    cols = len(grid_lines["vertical_lines"]) - 1
    rows = len(grid_lines["horizontal_lines"]) - 1

    for index, row in dataset_df.iterrows():
        col, row = find_grid_index(row['longitude'], row['latitude'], grid_lines, cols, rows)
        # sample_positions_on_grid[index, 0] = -1 if (row == -1 or col) else row * cols + col
        sample_positions_on_grid[index, 0] = row
        sample_positions_on_grid[index, 1] = col
        sample_positions_on_grid[index, 2] = -1 if (row == -1 or col == -1) else row * cols + col

    df = pd.DataFrame(sample_positions_on_grid, columns=["row", "col", "idx"])
    df = df.join(dataset_df[["longitude", "latitude"]])
    return df


def save(label_df, augmented_df, grid_lines, filename, save_path, grid_length):
    head, tail = os.path.split(filename)

    filename_list = tail.split(".")

    label_filename = filename_list[0] + "_gridlen" + str(grid_length) + "_label." + filename_list[1]
    augmented_filename = filename_list[0] + "_gridlen" + str(grid_length) + "." + filename_list[1]
    grid_lines_name = filename_list[0] + "_grid_lines_gridlen" + str(grid_length) + ".json"

    label_file_path = os.path.join(save_path, label_filename)
    augmented_file_path = os.path.join(save_path, augmented_filename)
    grid_lines_file_path = os.path.join(save_path, grid_lines_name)

    print(label_file_path)
    print(augmented_file_path)
    print(grid_lines_file_path)
    label_df.to_csv(label_file_path, index=False)

    # don't include latitude and longitude in training data
    augmented_df.drop(["longitude", "latitude"], axis=1).to_csv(augmented_file_path, index=False)
    with open(grid_lines_file_path, 'w') as fp:
        json.dump(grid_lines, fp)


def load(filename, save_path, grid_length):
    head, tail = os.path.split(filename)

    filename_list = tail.split(".")

    label_filename = filename_list[0] + "_gridlen" + str(grid_length) + "_label." + filename_list[1]
    data_filename = filename_list[0] + "_gridlen" + str(grid_length) + "." + filename_list[1]
    grid_lines_name = filename_list[0] + "_grid_lines_gridlen" + str(grid_length) + ".json"

    label_file_path = os.path.join(save_path, label_filename)
    data_file_path = os.path.join(save_path, data_filename)
    grid_lines_file_path = os.path.join(save_path, grid_lines_name)

    label_df = pd.read_csv(label_file_path)
    data_df = pd.read_csv(data_file_path)
    with open(grid_lines_file_path, 'r') as fp:
        grid_lines = json.load(fp)

    return label_df, data_df, grid_lines, label_file_path, data_file_path


def grid_index_to_coordinates(grid_lines, grid_indices):
    cols = len(grid_lines["vertical_lines"]) - 1
    rows = len(grid_lines["horizontal_lines"]) - 1

    coordinates = []

    for idx in grid_indices:
        row_idx = int(idx / cols)
        col_idx = idx % cols

        top_left_coor = grid_lines["vertical_lines"][col_idx]["start"]
        top_right_coor = grid_lines["vertical_lines"][col_idx + 1]["start"]

        left_top_coor = grid_lines["horizontal_lines"][row_idx]["start"]
        left_bottom_coor = grid_lines["horizontal_lines"][row_idx + 1]["start"]

        # change lat, lon order
        top_distance_meter, top_bearing_angle_deg = get_distance_and_bearing((top_left_coor[1], top_left_coor[0]),
                                                                             (top_right_coor[1], top_right_coor[0]))
        left_distance_meter, left_bearing_angle_deg = get_distance_and_bearing((left_top_coor[1], left_top_coor[0]), (
            left_bottom_coor[1], left_bottom_coor[0]))

        top_middle_coor = point_at((top_left_coor[1], top_left_coor[0]), top_distance_meter / 2, top_bearing_angle_deg)
        left_middle_coor = point_at((left_top_coor[1], left_top_coor[0]), left_distance_meter / 2,
                                    left_bearing_angle_deg)

        # use longitude of top line and latitude of left line
        longitude = top_middle_coor[1]
        latitude = left_middle_coor[0]
        coordinates.append([longitude, latitude])

    return pd.DataFrame(coordinates, columns=["longitude", "latitude"])


def probability_distribution_to_coordinates(grid_lines, grid_probabilities, grid_length, bearing_angle_deg, k):
    """

    :param grid_lines:
    :param grid_probabilities:
    :param grid_length:
    :param bearing_angle_deg:
    :param k: number of used probabilites in weighting
    :return:
    """
    grid_element_center_coordinates = calculate_grid_element_center_coordinates(grid_lines, grid_length, bearing_angle_deg)
    grid_element_center_coordinates_np = np.array(grid_element_center_coordinates)

    grid_element_center_coordinates_np = grid_element_center_coordinates_np.reshape((-1, 2))


    coordinates = []


    for prabability_distribution in grid_probabilities:
        probability_distribution_np = np.array(prabability_distribution).reshape((-1, 1))
        sorted_probabilites = np.sort(probability_distribution_np, axis=None)
        threshold = sorted_probabilites[-k]
        # print(sorted_probabilites)
        # normalize remaining probabilites
        probability_distribution_np[probability_distribution_np < threshold] = 0
        probability_distribution_np = probability_distribution_np / np.sum(probability_distribution_np)


        weighted_coordinates = probability_distribution_np * grid_element_center_coordinates_np

        # print(probability_distribution_np)
        average_coordinate = np.sum(weighted_coordinates, axis=0)
        # print(average_coordinate)
        # use longitude of top line and latitude of left line

        coordinates.append(average_coordinate)

    return pd.DataFrame(coordinates, columns=["longitude", "latitude"])


def get_prediction_label_distance_df(prediction_coordinates_df, label_coordinates_df):
    distances = np.zeros((len(prediction_coordinates_df), 1))

    for i in range(len(prediction_coordinates_df)):
        pred_row = prediction_coordinates_df.iloc[i]
        label_row = label_coordinates_df.iloc[i]

        pred_coor = (pred_row["latitude"], pred_row["longitude"])
        label_coor = (label_row["latitude"], label_row["longitude"])

        distance_m, bearing_deg = get_distance_and_bearing(pred_coor, label_coor)
        distances[i] = distance_m

    return pd.DataFrame(distances, columns=["distance"])


def remove_outliers(prediction_coordinates_df, label_coordinates_df, threshold, moving_average_bins):
    rows_to_drop = []

    # distances = np.zeros(len(prediction_coordinates_df)-1)
    # for i in range(len(prediction_coordinates_df)-1):
    #     cur_row = prediction_coordinates_df.iloc[i]
    #     cur_coordinate = np.array([cur_row["latitude"], cur_row["longitude"]])
    #
    #     next_row = prediction_coordinates_df.iloc[i+1]
    #     next_coordinate = np.array([next_row["latitude"], next_row["longitude"]])
    #     distance_m, bearing_deg = get_distance_and_bearing(cur_coordinate, next_coordinate)
    #     distances[i] = distance_m
    #
    # plt.plot(distances)
    # fig, ax = plt.subplots()
    # ax.plot(distances)
    #
    # st.pyplot(fig)
    #
    # outliers = distances > threshold
    # rows_to_drop = np.where(outliers)[0] +1
    # st.write(outliers)
    # st.write(rows_to_drop)

    for i in range(moving_average_bins, len(prediction_coordinates_df)):
        average_coordinate = 0  # normal average is enough since coordinates are close
        for j in range(moving_average_bins):
            current_idx = i - (j + 1)
            current_row = prediction_coordinates_df.iloc[current_idx]
            current_coor = np.array([current_row["latitude"], current_row["longitude"]])
            average_coordinate += current_coor

        average_coordinate /= moving_average_bins

        pred_row = prediction_coordinates_df.iloc[i]
        pred_coordinate = np.array([pred_row["latitude"], pred_row["longitude"]])
        distance_m, bearing_deg = get_distance_and_bearing(pred_coordinate, average_coordinate)

        if distance_m > threshold:
            rows_to_drop.append(i)

    return prediction_coordinates_df.drop(rows_to_drop).reset_index(drop=True), label_coordinates_df.drop(rows_to_drop).reset_index(drop=True)


def get_selected_model_predictions(model_name, grid_lines, use_probability_weighting, grid_length, bearing_angle_deg):
    prediction_grid_indices, label_grid_indices, prediction_probability_distributions = None, None, None

    if model_name == "mlp_9_grid100":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=300,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid100",
            output_classes=64,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen100.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen100_label.csv",
            batch_size=128,
            num_prev_steps=1)

    elif model_name == "mlp_9_grid100_prev5":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=20,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid100_prev5",
            output_classes=64,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen100.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen100_label.csv",
            batch_size=32,
            num_prev_steps=5)

    elif model_name == "mlp_9_grid100_prev10":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=65,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid100_prev10",
            output_classes=64,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen100.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen100_label.csv",
            batch_size=32,
            num_prev_steps=10)

    elif model_name == "mlp_9_grid50_prev1":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=60,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid50_prev1",
            output_classes=64 * 4,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen50.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen50_label.csv",
            batch_size=32,
            num_prev_steps=1)


    elif model_name == "mlp_9_grid50_prev5":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=29,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid50_prev5",
            output_classes=64 * 4,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen50.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen50_label.csv",
            batch_size=32,
            num_prev_steps=5)

    elif model_name == "mlp_9_grid50_prev10":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=29,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid50_prev10",
            output_classes=64 * 4,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen50.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen50_label.csv",
            batch_size=32,
            num_prev_steps=10)

    elif model_name == "mlp_9_grid20_prev1":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=65,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid20_prev1",
            output_classes=64 * 25,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen20.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen20_label.csv",
            batch_size=32,
            num_prev_steps=1)

    elif model_name == "mlp_9_grid20_prev5":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=21,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid20_prev5",
            output_classes=64 * 25,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen20.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen20_label.csv",
            batch_size=32,
            num_prev_steps=5)

    elif model_name == "mlp_9_grid20_prev10":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=67,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid20_prev10",
            output_classes=64 * 25,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen20.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen20_label.csv",
            batch_size=32,
            num_prev_steps=10)

    elif model_name == "mlp_9_grid10_prev1":

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=67,
            checkpoint_folder="./nn/checkpoints/mlp_9_grid10_prev1",
            output_classes=64 * 25 * 4,
            input_features=9,
            test_x_directory="./datasets/erlangen_test_dataset_gridlen10.csv",
            test_y_directory="./datasets/erlangen_test_dataset_gridlen10_label.csv",
            batch_size=32,
            num_prev_steps=1)

    elif model_name == "lstm_9_grid100_prev5":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=600,
            checkpoint_folder="./lstm/checkpoints/lstm_9_grid100_prev5",
            output_classes=64,
            input_features=9,
            x_directory="./datasets/erlangen_test_dataset_gridlen100.csv",
            y_directory="./datasets/erlangen_test_dataset_gridlen100_label.csv",
            batch_size=32,
            num_prev_steps=5)

    elif model_name == "lstm_9_grid100_prev10":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=900,
            checkpoint_folder="./lstm/checkpoints/lstm_9_grid100_prev10",
            output_classes=64,
            input_features=9,
            x_directory="./datasets/erlangen_test_dataset_gridlen100.csv",
            y_directory="./datasets/erlangen_test_dataset_gridlen100_label.csv",
            batch_size=32,
            num_prev_steps=10)

    elif model_name == "lstm_9_grid50_prev5":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=100,
            checkpoint_folder="./lstm/checkpoints/lstm_9_grid50_prev5",
            output_classes=64*4,
            input_features=9,
            x_directory="./datasets/erlangen_test_dataset_gridlen50.csv",
            y_directory="./datasets/erlangen_test_dataset_gridlen50_label.csv",
            batch_size=32,
            num_prev_steps=5)

    elif model_name == "lstm_9_grid50_prev10":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=820,
            checkpoint_folder="./lstm/checkpoints/lstm_9_grid50_prev10",
            output_classes=64*4,
            input_features=9,
            x_directory="./datasets/erlangen_test_dataset_gridlen50.csv",
            y_directory="./datasets/erlangen_test_dataset_gridlen50_label.csv",
            batch_size=32,
            num_prev_steps=10)

    elif model_name == "lstm_9_grid20_prev5":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=123,
            checkpoint_folder="./lstm/checkpoints/lstm_9_grid20_prev5",
            output_classes=64 * 25,
            input_features=9,
            x_directory="./datasets/erlangen_test_dataset_gridlen20.csv",
            y_directory="./datasets/erlangen_test_dataset_gridlen20_label.csv",
            batch_size=32,
            num_prev_steps=5)

    elif model_name == "lstm_9_grid20_prev10":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=123,
            checkpoint_folder="./lstm/checkpoints/lstm_9_grid20_prev10",
            output_classes=64 * 25,
            input_features=9,
            x_directory="./datasets/erlangen_test_dataset_gridlen20.csv",
            y_directory="./datasets/erlangen_test_dataset_gridlen20_label.csv",
            batch_size=32,
            num_prev_steps=10)



    elif model_name == "lstm_24_grid20_prev10":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = lstm.test.get_model_predictions_on_test_dataset(
            restored_checkpoint=490,
            checkpoint_folder="./lstm/checkpoints/lstm_24_grid20_prev10",
            output_classes=64 * 25,
            input_features=24,
            x_directory="./datasets/erlangen_test_dataset_minadjusted_gridlen20.csv",
            y_directory="./datasets/erlangen_test_dataset_minadjusted_gridlen20_label.csv",
            batch_size=32,
            num_prev_steps=10)

    elif model_name == "mlp_18_grid50_prev15_normalized":
        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn_no_preaugmentation.test.get_model_predictions_on_test_dataset(
                                                restored_checkpoint=100,
                                              checkpoint_folder="./nn_no_preaugmentation/checkpoints/mlp_18_grid50_prev15_normalized",
                                              output_classes=64 * 4,
                                              input_features=18,
                                              test_x_directory="./datasets/erlangen_test_dataset_gridlen50.csv",
                                              test_y_directory="./datasets/erlangen_test_dataset_gridlen50_label.csv",
                                              batch_size=32,
                                              num_prev_steps=15,
                                              train_x_directory="./datasets/erlangen_dataset_gridlen50.csv",
                                              train_y_directory="./datasets/erlangen_dataset_gridlen50_label.csv"
                                              )
    elif model_name == "mlp_9_grid50_prev3_normalized_minadjusted":

        # dataset parameters
        GRID_WIDTH = 800
        GRID_HEIGHT = 800
        grid_element_length = 50
        num_prev_steps = 3
        input_features = 9
        restored_checkpoint = 156
        normalize = True
        # dataset parameters

        output_classes = int((GRID_WIDTH / grid_element_length) * (GRID_HEIGHT / grid_element_length))
        network_input_length = num_prev_steps * input_features

        checkpoint_folder = f"./nn_no_preaugmentation/checkpoints/mlp_{input_features}_grid{grid_element_length}_prev{num_prev_steps}{'_normalized' if normalize else ''}_minadjusted"

        train_x_directory = f"./datasets/erlangen_dataset_minadjusted_gridlen{grid_element_length}.csv"
        train_y_directory = f"./datasets/erlangen_dataset_minadjusted_gridlen{grid_element_length}_label.csv"

        test_x_directory = f"./datasets/erlangen_test_dataset_minadjusted_gridlen{grid_element_length}.csv"
        test_y_directory = f"./datasets/erlangen_test_dataset_minadjusted_gridlen{grid_element_length}_label.csv"

        prediction_grid_indices, label_grid_indices, prediction_probability_distributions = nn_no_preaugmentation.test.get_model_predictions_on_test_dataset(restored_checkpoint=restored_checkpoint,
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

    elif model_name == "random_forest_grid20":
        @st.cache_resource
        def load_rf():
            rf = random_forest.random_forest.RandomForest(x_directory="./datasets/erlangen_dataset_gridlen20.csv",
                                                          y_directory="./datasets/erlangen_dataset_gridlen20_label.csv",
                                                          x_test_directory="./datasets/erlangen_test_dataset_gridlen20.csv",
                                                          y_test_directory="./datasets/erlangen_test_dataset_gridlen20_label.csv",
                                                          num_features=12)
            rf.fit()
            return rf
        rf = load_rf()
        prediction_grid_indices, label_grid_indices = rf.test()

    if use_probability_weighting is False:
        prediction_coordinates_df = grid_index_to_coordinates(grid_lines, prediction_grid_indices)
    else:
        prediction_coordinates_df = probability_distribution_to_coordinates(grid_lines, prediction_probability_distributions, grid_length, bearing_angle_deg, k=1)

    return prediction_coordinates_df


def draw_coordinates_at_selected_time(prediction_df, label_df, time_idx):
    layers = []

    prediction_coordinates_at_selected_time_df = prediction_df.iloc[time_idx].to_frame().T
    label_coordinates_at_selected_time_df = label_df.iloc[time_idx].to_frame().T


    prediction_coordinates_at_selected_time = pdk.Layer(
        "ScatterplotLayer",
        data=prediction_coordinates_at_selected_time_df,
        pickable=False,
        opacity=1,
        stroked=True,
        filled=False,
        line_width_min_pixels=1,
        get_position=["longitude", "latitude"],
        get_radius=10,
        radius_min_pixels=1,
        radiusScale=1,
        # radius_max_pixels=60,
        get_fill_color=[0, 252, 0],
        get_line_color=[0, 252, 0],
        tooltip="test test",
    )

    label_coordinates_at_selected_time = pdk.Layer(
        "ScatterplotLayer",
        data=label_coordinates_at_selected_time_df,
        pickable=False,
        opacity=1,
        stroked=True,
        filled=False,
        line_width_min_pixels=1,
        get_position=["longitude", "latitude"],
        get_radius=10,
        radius_min_pixels=1,
        radiusScale=1,
        # radius_max_pixels=60,
        get_fill_color=[252, 0, 0],
        get_line_color=[255, 0, 0],
        tooltip="test test",
    )
    layers.append(prediction_coordinates_at_selected_time)
    layers.append(label_coordinates_at_selected_time)
    return layers


def correct_offset(long_df, short_df):
    short_len = len(short_df)
    long_len = len(long_df)
    offset = long_len - short_len
    if offset > 0:
        offset_corrected_long_df = long_df.iloc[offset:].reset_index(drop=True)
        return offset_corrected_long_df

    return long_df

# End of Helper functions
