import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
plt.style.use('seaborn')
import time


pose_keypoints = np.array([0, 16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])  # Now includes nose


def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):
    """Now visualize in 3D"""
    torso = [[1, 7], [2, 8], [7, 8], [1, 2]]
    armr = [[1, 3], [3, 5]]
    arml = [[2, 4], [4, 6]]
    legr = [[7, 9], [9, 11]]
    legl = [[8, 10], [10, 12]]
    head = [[0, 1], [0, 2]]

    body = [torso, armr, arml, legr, legl, head]
    colors = ['red', 'blue', 'green', 'black', 'orange', 'yellow']

    # Swap X and Z axes, and negate Z
    p3ds_swapped = p3ds[:, :, [0, 2, 1]]
    p3ds_swapped[:, :, 2] *= -1

    # Filter valid keypoints for range calculation
    valid_points = p3ds_swapped.reshape(-1, 3)
    valid_points = valid_points[~np.any(valid_points == -1, axis=1)]
    min_vals = np.min(valid_points, axis=0)
    max_vals = np.max(valid_points, axis=0)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    time.sleep(5)

    for framenum, kpts3d in enumerate(p3ds_swapped):
        if framenum % 2 == 0:
            continue

        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
                        ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
                        zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]],
                        linewidth=4, c=part_color)

        for i, (x, y, z) in enumerate(kpts3d):
            if (x, y, z) != (-1, -1, -1):
                ax.text(x, y, z, str(i), color='black', fontsize=8)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Set limits based on all keypoints
        ax.set_xlim(-2,max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])

        plt.pause(1 / 30)

        if plt.waitforbuttonpress(timeout=0.01):
            print("Paused. Press any key to resume...")
            while not plt.waitforbuttonpress():
                pass

        ax.cla()



if __name__ == '__main__':

    p3ds = read_keypoints('3dpart/kpts_3d.dat')
    
    visualize_3d(p3ds)

