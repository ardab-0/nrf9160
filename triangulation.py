import numpy as np
import matplotlib.pyplot as plt

def triangulate(points_np, signal_strengths_np):

    total_strength = np.sum(signal_strengths_np)
    weights = signal_strengths_np / total_strength

    position = np.sum(points_np * weights, axis=1)
    return position



def test_triangulate():
    points = np.array([[1, 7, 1],
                       [2, 5, 6]])

    signal_strengths = np.array([20, 70, 30])
    pos = triangulate(points, signal_strengths)
    print(pos)

    plt.scatter(points[0, :], points[1, :], alpha=0.5)
    plt.scatter(pos[0], pos[1], c=2)
    for i, txt in enumerate(signal_strengths):
        plt.annotate("Strength: {}".format(txt), (points[0, i], points[1, i]))
    plt.annotate("Combined position", pos)
    plt.title("Triangulation of position based on signal strengths")
    plt.show()

test_triangulate()