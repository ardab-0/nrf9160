from geographiclib.geodesic import Geodesic
import numpy as np
import geopy.distance
from geopy import distance



def get_distance_and_bearing(point_from, point_to):
    """

    :param point_from: (latitude, longitude)
    :param point_to: (latitude, longitude)
    :return: distance(m), bearing (degrees)
    """
    # distance_meter_geopy = distance.distance(point_from, point_to).meters
    inverse_dict = Geodesic.WGS84.Inverse(point_from[0], point_from[1], point_to[0], point_to[1])
    distance_meter = inverse_dict["s12"]
    bearing_angle_deg = inverse_dict["azi1"]

    # print("Geopy distance: ", distance_meter_geopy)
    # print("Geodesic distance: ", distance_meter)

    return distance_meter, bearing_angle_deg




def get_cartesian_coordinates(coordinates_np):
    i = 1
    cartesian_coordinates_np = np.zeros_like(coordinates_np)

    while i < len(coordinates_np):
        distance_meter, bearing_angle_deg = get_distance_and_bearing(coordinates_np[i - 1],
                                                                     coordinates_np[i])
        # print("Distance:", distance_meter)
        # print("Angle", bearing_angle_deg)
        dx = distance_meter * np.cos(np.deg2rad(bearing_angle_deg))
        dy = distance_meter * np.sin(np.deg2rad(bearing_angle_deg))
        # print("dx:", dx)
        # print("dy", dy)
        # print("\n\n")
        cartesian_coordinates_np[i] = cartesian_coordinates_np[i - 1] + np.array([dx, dy])
        i += 1
    return cartesian_coordinates_np


def get_coordinates(cartesian_coordinates_np, original_coordinates_np):
    coordinates = np.zeros_like(original_coordinates_np)
    coordinates[0] = original_coordinates_np[0]
    i = 1

    while i < len(coordinates):
        displacement = cartesian_coordinates_np[i] - cartesian_coordinates_np[i - 1]
        dist_m = np.linalg.norm(displacement)
        bearing_angle_deg = np.rad2deg(np.arctan2(displacement[1], displacement[0]))

        next_point = geopy.distance.distance(meters=dist_m).destination((coordinates[i - 1, 0], coordinates[i - 1, 1]),
                                                                        bearing=bearing_angle_deg)
        coordinates[i, 0], coordinates[i, 1] = next_point[0], next_point[1]
        i += 1

    return coordinates

def point_at(source_point, distance_meter, angle_deg):
    next_point = geopy.distance.distance(meters=distance_meter).destination(source_point,
                                                                    bearing=angle_deg)
    return next_point


def get_distances_in_cartesian(distances, positions, positions_cartesian_coordinates):
    average_position_cartesian = np.mean(positions_cartesian_coordinates, axis=0)
    distances_cartesian = []

    for i, distance in enumerate(distances.T):
        displacement_cartesian = average_position_cartesian - positions_cartesian_coordinates[i, :]
        bearing_angle_deg = np.rad2deg(np.arctan2(displacement_cartesian[1], displacement_cartesian[0]))

        center_in_geological = geopy.distance.distance(meters=distance).destination((positions[i, 0], positions[i, 1]), bearing=bearing_angle_deg)

        positions_with_center = np.zeros((positions.shape[0] + 1, positions.shape[1]))
        positions_with_center[:positions.shape[0], :] = positions
        positions_with_center[-1, :] = np.array([center_in_geological[0], center_in_geological[1]])

        positions_with_center_cartesian = get_cartesian_coordinates(positions_with_center)

        center_cartesian = positions_with_center_cartesian[-1, :]
        distance_cartesian = np.linalg.norm(positions_cartesian_coordinates[i, :] - center_cartesian)
        distances_cartesian.append(distance_cartesian)

    return np.array(distances_cartesian)


