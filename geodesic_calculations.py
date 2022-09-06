from geographiclib.geodesic import Geodesic
import numpy as np


def get_distance_and_bearing(point_from, point_to):
    """

    :param point_from: (latitude, longitude)
    :param point_to: (latitude, longitude)
    :return: distance(m), bearing (degrees clockwise from north)
    """
    inverse_dict = Geodesic.WGS84.Inverse(point_from[0], point_from[1], point_to[0], point_to[1])
    distance_meter = inverse_dict["s12"]
    bearing_angle_deg = inverse_dict["azi1"]
    return distance_meter, bearing_angle_deg


def adjust_angle(angle):
    return - angle + 90


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
        i+=1
    return cartesian_coordinates_np
