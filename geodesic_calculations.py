from geographiclib.geodesic import Geodesic


inverse_dict = Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)
distance_meter = inverse_dict["s12"]

# (bearing) is given by azi1 (161.067... degrees clockwise from north)
bearing_angle_deg = inverse_dict["azi1"]
print(inverse_dict)
print(distance_meter)
print(bearing_angle_deg)