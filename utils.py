import pandas as pd
import streamlit as st
import numpy as np
from constants import frequency_upperband_dict, frequency_lowerband_dict

def construct_measurement_dictionary(measurement_data, return_measurement_list=False):
    measurement_list = measurement_data.split(",")
    current_params = ["status", "cell_id", "plmn", "tac", "timing_advance", "current_earfcn",
                      "current_phys_cell_id", "current_rsrp", "current_rsrq", "measurement_time"]
    neighbor_params = ["n_earfcn", "n_phys_cell_id", "n_rsrp", "n_rsrq", "time_diff"]

    measurement_dict = {}

    no_of_neighbors = int((len(measurement_list) - 1 - len(current_params)) / len(neighbor_params))

    for i, param in enumerate(current_params):
        measurement_dict[param] = measurement_list[i].replace('"', '')

    if no_of_neighbors > 0:
        measurement_dict["neighbor_cells"] = []

    for i in range(no_of_neighbors):
        neighbor_dict = {}
        for j, param in enumerate(neighbor_params):
            neighbor_dict[param] = measurement_list[len(current_params) + len(neighbor_params) * i + j].replace('"', '')

        measurement_dict["neighbor_cells"].append(neighbor_dict)

    measurement_dict["timing_advance_measurement_time"] = measurement_list[-1]
    if return_measurement_list:
        return measurement_dict, measurement_list
    return measurement_dict


def get_measurement_dictionary_list(measurement_results):
    measurement_dictionary_list = []
    for measurement_list in measurement_results:
        measurement_dictionary_list.append(construct_measurement_dictionary(measurement_list))

    return measurement_dictionary_list


@st.cache(allow_output_mutation=True)
def load_data(dataset_file):
    base_station_df = pd.read_csv(dataset_file)
    base_station_df.columns = ["Radio", "MCC", "MNC", "TAC", "CID", "Unit", "Longitude",
                               "Latitude", "Range", "Samples",
                               "Changeable=1", "Created", "Updated", "AverageSignal"]
    return base_station_df


def query_base_station_dataset(df, plmn, tac, cell_id):
    plmn_int = int(plmn)
    tac_decimal = int(tac, 16)
    cell_id_decimal = int(cell_id, 16)
    mcc = 0
    mnc = 0

    if len(plmn) == 6:
        mcc = int(plmn_int / 1000)
        mnc = plmn_int % 1000
    elif len(plmn) == 5:
        mcc = int(plmn_int / 100)
        mnc = plmn_int % 100

    query_results = df.loc[
        (df["MCC"] == mcc) & (df["MNC"] == mnc) & (df["TAC"] == tac_decimal) & (
                df["CID"] == cell_id_decimal)]

    return query_results.head(1)


def get_moving_path_df(base_station_df, moving_measurement_dictionary_list):
    df = pd.DataFrame()
    for dictionary in moving_measurement_dictionary_list:
        res = query_base_station_dataset(base_station_df, dictionary["plmn"],
                                         dictionary["tac"], dictionary["cell_id"])
        res["measurement_time"] = int(dictionary["measurement_time"])
        res["current_rsrp"] = int(dictionary["current_rsrp"])
        res["current_rsrq"] = int(dictionary["current_rsrq"])
        df = pd.concat([df, res])
    return df


def get_kalman_matrices(measurement_sigma=1, dt=1, sigma_a=1):
    F = np.array([[1, dt, 0.5 * dt ** 2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, 0.5 * dt ** 2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]
                  ], dtype=float)

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=float)

    R = np.array([[measurement_sigma, 0],
                  [0, measurement_sigma]], dtype=float)

    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2, 0, 0, 0],
                                 [dt ** 3 / 2, dt ** 2, dt, 0, 0, 0],
                                 [dt ** 2 / 2, dt, 1, 0, 0, 0],
                                 [0, 0, 0, dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                                 [0, 0, 0, dt ** 3 / 2, dt ** 2, dt],
                                 [0, 0, 0, dt ** 2 / 2, dt, 1]], dtype=float)

    return F, H, R, Q







def get_band(earfcn):
    for band in frequency_lowerband_dict:
        if earfcn >= frequency_lowerband_dict[band] and earfcn < frequency_upperband_dict[band]:
            return int(band)

