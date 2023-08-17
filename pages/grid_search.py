import streamlit as st
import pydeck as pdk
import pandas as pd
from nn.data_augmentation import one_to_many_augmenter
from geodesic_calculations import point_at, get_distance_and_bearing
import glob
import grid_operations.grid_operations as gridops
from constants import BEARING_ANGLE_DEG, GRID_PAGE_DATASET_PATH, DATASET_PATH


combined_measurement_filenames = glob.glob(GRID_PAGE_DATASET_PATH + "*.csv")


layers_to_plot = []
selected_combined_measurement_filename = st.sidebar.selectbox("Select dataset file to load", combined_measurement_filenames)

df = pd.read_csv(selected_combined_measurement_filename)

st.write("Selected Dataset")
st.write(df)

# Initial coordinates for sliders
lon = 11.02860602
lat = 49.57246557
lon_max = lon + 0.01
lon_min = lon - 0.01
lat_max = lat + 0.01
lat_min = lat - 0.01
# Initial coordinates


grid_length = st.sidebar.select_slider('Grid Length (m)', options=[20, 40, 50])


label_df, augmented_df, grid_lines, label_file_path, augmented_file_path = gridops.load(selected_combined_measurement_filename,
                                                                                        DATASET_PATH, grid_length, is_minadjusted=True)

cols = len(grid_lines["vertical_lines"]) - 1
rows = len(grid_lines["horizontal_lines"]) - 1
output_classes = cols * rows


selected_device_type = st.sidebar.selectbox('Use CPU / GPU', ["cpu", "cuda"])
use_cuda = False if selected_device_type == "cpu" else True

num_prev_steps = st.sidebar.slider('Number of previous steps: ', 1, 10)
input_features = st.sidebar.slider('Number of input features: ', 6, 18, step=3)

prediction_coordinates_df = gridops.get_selected_grid_search_model_predictions(num_prev_steps,input_features, grid_lines,
                                                                   use_probability_weighting=False, grid_length=grid_length, bearing_angle_deg=BEARING_ANGLE_DEG, use_cuda=use_cuda,
                                                                               output_classes=output_classes, grid_element_length=grid_length)

label_coordinates_df = label_df[["latitude", "longitude"]]

offset_corrected_label_coordinates_df = gridops.correct_offset(label_coordinates_df, prediction_coordinates_df)

if st.sidebar.checkbox("Remove Outliers"):
    prediction_coordinates_df, offset_corrected_label_coordinates_df = gridops.remove_outliers(
        prediction_coordinates_df, offset_corrected_label_coordinates_df, 30, 3)

distance_df = gridops.get_prediction_label_distance_df(prediction_coordinates_df,
                                                       offset_corrected_label_coordinates_df)

# print(distance_df)
pred_label_df = pd.concat([prediction_coordinates_df, offset_corrected_label_coordinates_df], axis=1)
pred_label_df.columns = ["prediction_longitude", "prediction_latitude", "label_longitude", "label_latitude"]
pred_label_df = pred_label_df.join(distance_df)

st.write("Predictions, Offset Corrected Labels and Distances(m)")
st.write(pred_label_df)
st.write("Model Type: LSTM")
st.write("Mean distance between predictions and labels (m): ", pred_label_df["distance"].mean())

index_of_selected_estimation_result = st.slider("Select time index", 0, len(prediction_coordinates_df) - 1, value=0)
layers_to_plot.extend(
    gridops.draw_coordinates_at_selected_time(prediction_coordinates_df, offset_corrected_label_coordinates_df,
                                              index_of_selected_estimation_result))

prediction_positions = pdk.Layer(
    "ScatterplotLayer",
    data=prediction_coordinates_df,
    pickable=False,
    opacity=1,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius=1,
    radius_min_pixels=1,
    radiusScale=1,
    # radius_max_pixels=60,
    get_fill_color=[0, 255, 0],
    get_line_color=[0, 255, 0],
    tooltip="test test",
)

layers_to_plot.append(prediction_positions)

label_positions = pdk.Layer(
    "ScatterplotLayer",
    data=label_coordinates_df,
    pickable=False,
    opacity=1,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius=1,
    radius_min_pixels=1,
    radiusScale=1,
    # radius_max_pixels=60,
    get_fill_color=[255, 0, 0],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

layers_to_plot.append(label_positions)



# outer_line_layer = pdk.Layer(
#     "LineLayer",
#     lines,
#     get_source_position="start",
#     get_target_position="end",
#     get_width=3,
#     get_color=[0, 0, 255],
#     pickable=False,
# )

grid_line_layer = pdk.Layer(
    "LineLayer",
    grid_lines["vertical_lines"] + grid_lines["horizontal_lines"],
    get_source_position="start",
    get_target_position="end",
    get_width=1,
    get_color=[0, 0, 255],
    pickable=False,
)

layers_to_plot.append(grid_line_layer)

view = pdk.ViewState(latitude=49.5, longitude=11, zoom=10, )
# Create the deck.gl map
r = pdk.Deck(
    layers=layers_to_plot,
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
map = st.pydeck_chart(r)
