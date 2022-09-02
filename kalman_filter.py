from filterpy.kalman import predict, update
import numpy as np
import matplotlib.pyplot as plt

dt = 1
sigma_a = 0.2


F = np.array([[1, dt, 0.5 * dt ** 2, 0, 0, 0],
              [0, 1, dt, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5 * dt ** 2],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]
              ], dtype=float)

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]], dtype=float)

R = np.array([[9, 0],
              [0, 9]], dtype=float)

Q = sigma_a**2 * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2, 0, 0, 0],
              [dt**3 / 2, dt**2, dt, 0, 0, 0],
              [dt**2 / 2, dt, 1, 0, 0, 0],
              [0, 0, 0, dt**4 / 4, dt**3 / 2, dt**2 / 2],
              [0, 0, 0, dt**3 / 2, dt**2, dt],
              [0, 0, 0, dt**2 / 2, dt, 1]], dtype=float)

P = np.eye(6) * 500

measurements = np.array([[-393.66, -375.93, -351.04, -328.96, -299.35, -273.36, -245.89, -222.58, -198.03, -174.17,
                          -146.32, -123.72, -103.47, -78.23, -52.63, -23.34, 25.96, 49.72, 76.94, 95.38, 119.83, 144.01,
                          161.84, 180.56, 201.42, 222.62, 239.4, 252.51, 266.26, 271.75, 277.4, 294.12, 301.23, 291.8,
                          299.89],
                         [300.4, 301.78, 295.1, 305.19, 301.06, 302.05, 300, 303.57, 296.33, 297.65, 297.41, 299.61,
                          299.6, 302.39, 295.04, 300.09, 294.72, 298.61, 294.64, 284.88, 272.82, 264.93, 251.46, 241.27,
                          222.98, 203.73, 184.1, 166.12, 138.71, 119.71, 100.41, 79.76, 50.62, 32.99, 2.14]]).T

x = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).T

filtered_measurements = np.zeros_like(measurements)

for i, z in enumerate(measurements):
    x, P = predict(x, P, F, Q)
    x, P = update(x, P, z, R, H)
    filtered_measurements[i] = np.squeeze(H @ x)


plt.scatter(measurements[:, 0], measurements[:, 1], alpha=0.5)
plt.scatter(filtered_measurements[:, 0], filtered_measurements[:, 1], alpha=0.5)
plt.title("Kalman Filter")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()