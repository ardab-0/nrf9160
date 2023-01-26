from single_measurement.threaded_serial import File_Reader_Writer, Serial_Communication
import pandas as pd
import numpy as np
from utils import construct_measurement_dictionary

measurement_filename = "../saved_measurements/measurements_erlangen_walking.json"
coordinate_filename = "../saved_measurements/20230124-145846 - Erlangen moving.kml"
dataset_filename = "../saved_measurements/erlangen_dataset.csv"


file_reader_writer = File_Reader_Writer(measurement_filename)
measurements, orig_position = file_reader_writer.read(get_orig_pos=True)


# "current_phys_cell_id", "current_rsrp", "current_rsrq", "measurement_time"
# "n_phys_cell_id", "n_rsrp", "n_rsrq"

max_neighbor_count = 0
for measurement_batch in measurements:
    measurement = measurement_batch[0]
    dictionary = construct_measurement_dictionary(measurement)
    if "neighbor_cells" in dictionary and len(dictionary["neighbor_cells"]) > max_neighbor_count:
        max_neighbor_count = len(dictionary["neighbor_cells"])

measurement_data_np = np.zeros((len(measurements), 4+max_neighbor_count*3 + 2 ))
column_labels = ["current_phys_cell_id", "current_rsrp", "current_rsrq", "measurement_time"]
for i in range(max_neighbor_count):
    column_labels.append(str(i+1)+"_phys_cell_id")
    column_labels.append(str(i + 1) + "_rsrp")
    column_labels.append(str(i + 1) + "n_rsrq")

column_labels.append("longitude")
column_labels.append("latitude")


with open(coordinate_filename, "r") as file1:
    file_content = file1.read()
    start = file_content.find("<coordinates>")
    end = file_content.find("</coordinates>")
    coordinate_text = file_content[start+14:end]
    coordinate_list = coordinate_text.split("\n")


for i, measurement_batch in enumerate(measurements):
    measurement = measurement_batch[0]
    dictionary = construct_measurement_dictionary(measurement)

    time_idx = int(round(int(dictionary["measurement_time"])/1000))
    measurement_data_np[i, :4] = np.array([dictionary["current_phys_cell_id"],
                                          dictionary["current_rsrp"],
                                          dictionary["current_rsrq"],
                                          time_idx
                                          ])
    if "neighbor_cells" in dictionary:
        for j, neighbor in enumerate(dictionary["neighbor_cells"]):
            measurement_data_np[i, 4+3*j:4+3*j+3] = np.array([neighbor["n_phys_cell_id"],
                                                              neighbor["n_rsrp"],
                                                              neighbor["n_rsrq"]])

    coordinates = coordinate_list[i].split(",")
    measurement_data_np[i, -2] = float(coordinates[0])
    measurement_data_np[i, -1] = float(coordinates[1])

measurement_data_df = pd.DataFrame(measurement_data_np, columns=column_labels)
measurement_data_df.to_csv(dataset_filename)

print("hi")