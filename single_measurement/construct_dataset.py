from single_measurement.threaded_serial import File_Reader_Writer, Serial_Communication
import pandas as pd
import numpy as np
from utils import construct_measurement_dictionary

# File names

measurement_filename = "../raw_measurements/Erlangen-15-02-2023-test.json"  # file produced by the capture_measurement.py
coordinate_filename = "../raw_measurements/Erlangen-15-02-2023-test.kml"  # file recorded by the gps tracker application
dataset_filename = "../combined_measurements/Erlangen-15-02-2023-test-minadjusted.csv"  # output file

# File names

max_values = {
    "physical_cell_id": 503,
    "rsrp": 255,
    "rsrq": 255
}

min_values = {
    "physical_cell_id": 0,
    "rsrp": -17,
    "rsrq": -30
}
def main():
    min_value_np = np.array([min_values["physical_cell_id"], min_values["rsrp"], min_values["rsrq"]])

    file_reader_writer = File_Reader_Writer(measurement_filename)
    measurements = file_reader_writer.read(get_orig_pos=False)

    main_base_station_columns = ["current_phys_cell_id", "current_rsrp", "current_rsrq"]
    neighbor_base_station_columns = ["n_phys_cell_id", "n_rsrp", "n_rsrq"]
    lat_lon = 2

    max_neighbor_count = 0
    for measurement_batch in measurements:
        measurement = measurement_batch[0]
        dictionary = construct_measurement_dictionary(measurement)
        if "neighbor_cells" in dictionary and len(dictionary["neighbor_cells"]) > max_neighbor_count:
            max_neighbor_count = len(dictionary["neighbor_cells"])

    measurement_data_np = np.ones((len(measurements), len(main_base_station_columns) + max_neighbor_count * len(
        neighbor_base_station_columns) + lat_lon)) * -1  # if there are missing values, th ey will be represented by -1

    column_labels = ["current_phys_cell_id", "current_rsrp", "current_rsrq"]
    for i in range(max_neighbor_count):
        column_labels.append(str(i + 1) + "_phys_cell_id")
        column_labels.append(str(i + 1) + "_rsrp")
        column_labels.append(str(i + 1) + "_rsrq")

    column_labels.append("longitude")
    column_labels.append("latitude")

    with open(coordinate_filename, "r") as file1:
        file_content = file1.read()
        start = file_content.find("<coordinates>")
        end = file_content.find("</coordinates>")
        coordinate_text = file_content[start + 14:end]
        coordinate_list = coordinate_text.split("\n")
        print("GPS Coordinate Recording Length (s): ", len(coordinate_list))

    for i, measurement_batch in enumerate(measurements):
        measurement = measurement_batch[0]
        dictionary = construct_measurement_dictionary(measurement)

        time_idx = int(round(int(dictionary["measurement_time"]) / 1000))
        measurement_data_np[i, :3] = np.array([dictionary["current_phys_cell_id"],
                                               dictionary["current_rsrp"],
                                               dictionary["current_rsrq"]
                                               ], dtype=int) - min_value_np
        if "neighbor_cells" in dictionary:
            for j, neighbor in enumerate(dictionary["neighbor_cells"]):
                measurement_data_np[i, len(main_base_station_columns) + len(neighbor_base_station_columns) * j:len(
                    main_base_station_columns) + len(neighbor_base_station_columns) * j + len(
                    neighbor_base_station_columns)] = np.array([neighbor["n_phys_cell_id"],
                                                                neighbor["n_rsrp"],
                                                                neighbor["n_rsrq"]], dtype=int) - min_value_np

        coordinates = coordinate_list[time_idx].split(",")
        measurement_data_np[i, -2] = float(coordinates[0])
        measurement_data_np[i, -1] = float(coordinates[1])

    measurement_data_df = pd.DataFrame(measurement_data_np, columns=column_labels)
    measurement_data_df.to_csv(dataset_filename, index=False)
    print("Dataset generated as: ", dataset_filename)


if __name__ == "__main__":
    main()
