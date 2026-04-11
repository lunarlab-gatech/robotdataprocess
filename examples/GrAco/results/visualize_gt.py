import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_txt(file_path):
    data = np.loadtxt(file_path)
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    return t, x, y


def load_all_trajectories(base_dir):
    trajs = []

    # find all ground* directories
    dirs = sorted(glob.glob(os.path.join(base_dir, "ground*")))

    for d in dirs:
        name = os.path.basename(d)  # e.g., ground4
        file_path = os.path.join(d, f"{name}.txt")

        if os.path.exists(file_path):
            t, x, y = load_txt(file_path)
            trajs.append((name, t, x, y))
        else:
            print(f"⚠️ Missing file: {file_path}")

    return trajs


def plot_all(base_dir):
    trajs = load_all_trajectories(base_dir)

    plt.figure(figsize=(8, 6))

    for name, t, x, y in trajs:
        plt.plot(x, y, label=name)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("All Ground Truth Trajectories")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    plt.show()


def animate_all(base_dir, step=100):
    trajs = load_all_trajectories(base_dir)

    min_len = min(len(x) for _, _, x, _ in trajs)

    plt.figure(figsize=(8, 6))

    for i in range(1, min_len, step):
        plt.clf()

        for name, t, x, y in trajs:
            plt.plot(x[:i], y[:i], label=name)
            plt.scatter(x[i-1], y[i-1], s=20)

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Trajectory Growth (step {i})")
        plt.legend()
        plt.grid()
        plt.axis('equal')

        plt.pause(0.01)

    plt.show()


if __name__ == "__main__":
    base_dir = "/media/nisemono/T7/GT/SLAM/Data/GrAco_dataset/V1.0"

    plot_all(base_dir)
    animate_all(base_dir, step=1000)