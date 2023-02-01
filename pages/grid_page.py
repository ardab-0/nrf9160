import numpy as np
import streamlit as st
import pydeck as pdk
import pandas as pd
from nn.data_augmentation import one_to_many_augmenter
from geodesic_calculations import point_at

dataset_filename = "./saved_measurements/erlangen_dataset.csv"
df = pd.read_csv(dataset_filename)

df = one_to_many_augmenter(df, distance_m=3, k=4)

st.write(df)

lon = 11.02860602
lat = 49.57246557
lon_max = lon + 0.01
lon_min = lon - 0.01
lat_max = lat + 0.01
lat_min = lat - 0.01


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


def calculate_grid_lines(horizontal_steps_top, vertical_steps_left, horizontal_steps_bottom, vertical_steps_right):
    lines = []

    for i in range(len(horizontal_steps_top)):
        lines.append({"start": horizontal_steps_top[i], "end": horizontal_steps_bottom[i]})

    for i in range(len(vertical_steps_left)):
        lines.append({"start": vertical_steps_left[i], "end": vertical_steps_right[i]})

    return lines, len(horizontal_steps_top) - 1, len(
        vertical_steps_left) - 1  # coordinate number is 1 more than the grid element number


def find_grid_index(lon, lat, grid_lines, cols, rows):
    row = -1
    col = -1
    is_any_inside_grid = False

    i = 0
    j = 0
    while i < cols:
        if (lon > grid_lines[i]["start"][0] and lon <= grid_lines[i + 1]["start"][0]):
            col = i
        i += 1

    while j < rows:
        if (lat < grid_lines[cols + 1 + j]["start"][1] and lat >= grid_lines[cols + 1 + j + 1]["start"][1]):
            row = j
        j += 1

    # if col == -1 or row == -1:
    #     col = -1
    #     row = -1

    return col, row


def divide_samples(dataset_df, grid_lines, cols, rows):
    sample_positions_on_grid = np.zeros((len(dataset_df), 3), dtype=int)

    for index, row in dataset_df.iterrows():
        col, row = find_grid_index(row['longitude'], row['latitude'], grid_lines, cols, rows)
        # sample_positions_on_grid[index, 0] = -1 if (row == -1 or col) else row * cols + col
        sample_positions_on_grid[index, 0] = row
        sample_positions_on_grid[index, 1] = col
        sample_positions_on_grid[index, 2] = -1 if (row == -1 or col == -1) else row * cols + col
    return pd.DataFrame(sample_positions_on_grid, columns=["row", "col", "idx"])


def save(df, filename, extension="csv"):
    filename_list = filename.split(".")
    label_filename = "."
    for el in filename_list:
        if el != extension:
            label_filename += el
        else:
            label_filename += "_label." + el
    print(label_filename)
    df.to_csv(label_filename, index=False)
################################# Sliders ############################################################

grid_length = st.sidebar.slider('Grid Length', 0, 1000, 100, step=10)

tl_lon = st.sidebar.slider('Top Left Longitude', lon_min, lon_max, 11.02860602, step=0.0001)
tl_lat = st.sidebar.slider('Top Left Latitude', lat_min, lat_max, 49.57246557, step=0.0001)

t_length = st.sidebar.slider('Top Edge', 0, 10000, grid_length, step=grid_length)
l_length = st.sidebar.slider('Left Edge', 0, 10000, grid_length, step=grid_length)
bearing_angle_deg = st.sidebar.slider('Bearing Angle Degree', -180., 180., 90., step=0.01)

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

grid_lines, cols, rows = calculate_grid_lines(horizontal_steps_top, vertical_steps_left, horizontal_steps_bottom,
                                              vertical_steps_right)
st.write(grid_lines)

grid_pos_idx_df = divide_samples(df, grid_lines, cols, rows)
st.write(grid_pos_idx_df)
if -1 in grid_pos_idx_df["idx"].values:
    st.write("Not all  points are inside grid")


if st.sidebar.button("Save label df"):
    save(grid_pos_idx_df, dataset_filename, extension="csv")


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

outer_line_layer = pdk.Layer(
    "LineLayer",
    lines,
    get_source_position="start",
    get_target_position="end",
    get_width=3,
    get_color=[0, 0, 255],
    pickable=False,
)

grid_line_layer = pdk.Layer(
    "LineLayer",
    grid_lines,
    get_source_position="start",
    get_target_position="end",
    get_width=1,
    get_color=[0, 0, 255],
    pickable=False,
)

view = pdk.ViewState(latitude=49.5, longitude=11, zoom=10, )
# Create the deck.gl map
r = pdk.Deck(
    layers=[gps_positions, outer_line_layer, grid_line_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
map = st.pydeck_chart(r)
