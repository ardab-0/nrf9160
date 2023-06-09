import streamlit as st
import pydeck as pdk
import pandas as pd
from nn.data_augmentation import one_to_many_augmenter
from geodesic_calculations import point_at, get_distance_and_bearing
import glob
import grid_operations.grid_operations as gridops

dataset_filenames = glob.glob("./combined_measurements/*.csv")
save_location_path = "./datasets/"  # save location of dataset which is divided into grid cells
MODELS = ["mlp_9_grid100", "mlp_9_grid100_prev5", "mlp_9_grid100_prev10",
          "mlp_9_grid50_prev1", "mlp_9_grid50_prev5", "mlp_9_grid50_prev10",
          "mlp_9_grid20_prev1", "mlp_9_grid20_prev5", "mlp_9_grid20_prev10",
          "mlp_9_grid10_prev1",
          "lstm_9_grid100_prev5", "lstm_9_grid100_prev10",
          "lstm_9_grid50_prev5", "lstm_9_grid50_prev10",
          "lstm_9_grid20_prev5", "lstm_9_grid20_prev10",
          "lstm_24_grid20_prev10",
          "random_forest_grid20", "mlp_18_grid50_prev15_normalized", "mlp_9_grid50_prev3_normalized_minadjusted"]
layers_to_plot = []
dataset_filename = st.sidebar.selectbox("Select file to load", dataset_filenames)
bearing_angle_deg = 90
df = pd.read_csv(dataset_filename)

st.write("Loaded Data Frame")
st.write(df)

# Initial coordinates for sliders
lon = 11.02860602
lat = 49.57246557
lon_max = lon + 0.01
lon_min = lon - 0.01
lat_max = lat + 0.01
lat_min = lat - 0.01
# Initial coordinates


grid_length = st.sidebar.slider('Grid Length', 0, 1000, 100, step=10)
mode = st.sidebar.radio("Select Mode", ('Grid Adjustment', 'Inference'))

if mode == "Inference":
    label_df, augmented_df, grid_lines, label_file_path, augmented_file_path = load(dataset_filename,
                                                                                    save_location_path, grid_length)

    cols = len(grid_lines["vertical_lines"]) - 1
    rows = len(grid_lines["horizontal_lines"]) - 1
    output_classes = cols * rows

    selected_model_name = st.selectbox('Select model type', MODELS)

    prediction_coordinates_df = gridops.get_selected_model_predictions(selected_model_name, grid_lines,
                                                                       use_probability_weighting=False)

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
    st.write("Mean distance (m): ", pred_label_df["distance"].mean())

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


else:  # Mode : Grid adjustment

    ################################# Sliders ############################################################

    k = st.sidebar.slider('Data Augmentation ', 0, 8, 0, step=1)

    tl_lon = st.sidebar.slider('Top Left Longitude', lon_min, lon_max, 11.02860602, step=0.0001)
    tl_lat = st.sidebar.slider('Top Left Latitude', lat_min, lat_max, 49.57246557, step=0.0001)

    t_length = st.sidebar.slider('Top Edge', 0, 10000, grid_length, step=grid_length)
    l_length = st.sidebar.slider('Left Edge', 0, 10000, grid_length, step=grid_length)

    ######################################################################################################

    tr = point_at((tl_lat, tl_lon), t_length, bearing_angle_deg)
    tr_lat, tr_lon = tr[0], tr[1]

    bl = point_at((tl_lat, tl_lon), l_length, bearing_angle_deg + 90)
    bl_lat, bl_lon = bl[0], bl[1]

    br = point_at((tr_lat, tr_lon), l_length, bearing_angle_deg + 90)
    br_lat, br_lon = br[0], br[1]

    lines = [{"start": [tl_lon, tl_lat], "end": [tr_lon, tr_lat]},
             {"start": [tr_lon, tr_lat], "end": [br_lon, br_lat]},
             {"start": [br_lon, br_lat], "end": [bl_lon, bl_lat]},
             {"start": [bl_lon, bl_lat], "end": [tl_lon, tl_lat]}]

    grid_lines, cols, rows = gridops.calculate_grid_lines(tl_lon, tl_lat, tr_lon, tr_lat, bl_lon, bl_lat, grid_length,
                                                          t_length,
                                                          l_length, bearing_angle_deg)
    # st.write(grid_lines)

    df = one_to_many_augmenter(df, distance_m=3, k=k)

    grid_pos_idx_df = gridops.divide_samples(df, grid_lines)
    st.write("Grid position indices")
    st.write(grid_pos_idx_df)
    if -1 in grid_pos_idx_df["idx"].values:
        st.write("Not all  points are inside grid")

    if st.sidebar.button("Save"):
        gridops.save(grid_pos_idx_df, df, grid_lines, dataset_filename, save_location_path, grid_length)

    gps_positions = pdk.Layer(
        "ScatterplotLayer",
        data=df,
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
        get_fill_color=[252, 0, 0],
        get_line_color=[255, 0, 0],
        tooltip="test test",
    )

    layers_to_plot.append(gps_positions)

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
