import cv2 as cv
import numpy as np
import glob
import os
from scipy import linalg

def load_calibration(filename):
    data = np.load(filename)
    return data['mtx'], data['dist']

def stereo_calibrate(mtx1, dist1, mtx2, dist2, left_folder, right_folder,
                     checkerboard_size=(7, 10), world_scaling=0.025):
    """
    Perform stereo calibration and return rotation and translation between cameras.
    """
    c1_images_names = sorted(glob.glob(os.path.join(left_folder, '*.jpg')))
    c2_images_names = sorted(glob.glob(os.path.join(right_folder, '*.jpg')))

    c1_images = [cv.imread(name, 1) for name in c1_images_names]
    c2_images = [cv.imread(name, 1) for name in c2_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    rows, cols = checkerboard_size

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * world_scaling

    imgpoints_left, imgpoints_right, objpoints = [], [], []

    for img1, img2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        ret1, corners1 = cv.findChessboardCorners(gray1, (rows, cols), None)
        ret2, corners2 = cv.findChessboardCorners(gray2, (rows, cols), None)

        if ret1 and ret2:
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            objpoints.append(objp)

    ret, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx1, dist1, mtx2, dist2,
        c1_images[0].shape[1::-1],
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC
    )

    print(f"Stereo calibration completed. Reprojection Error: {ret}")
    os.makedirs('CalibrationData', exist_ok=True)
    np.savez('CalibrationData/stereoExtrinsics.npz', R=R, T=T)
    return R, T

def get_projection_matrices(mtx1, mtx2, R, T):
    """
    Construct stereo projection matrices P1 and P2.
    """
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    RT2 = np.hstack((R, T))
    P1 = mtx1 @ RT1
    P2 = mtx2 @ RT2
    return P1, P2


def save_extrinsics(file_path, R, T):
    """
    Save rotation and translation matrices to a .dat file.
    """
    with open(file_path, 'w') as f:
        f.write("R:\n")
        for row in R:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("T:\n")
        for val in T.flatten():
            f.write(str(val) + "\n")

def stereo_pipeline(
    calib_path1='CalibrationData/c0_calib_data.npz',
    calib_path2='CalibrationData/c1_calib_data.npz',
    left_folder='left_calib',
    right_folder='right_calib',
    checkerboard_size=(7, 10),
    world_scaling=0.025
):
    """
    Complete stereo calibration and extrinsics saving pipeline.
    """
    mtx1, dist1 = load_calibration(calib_path1)
    mtx2, dist2 = load_calibration(calib_path2)

    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, left_folder, right_folder, checkerboard_size, world_scaling)
    P1, P2 = get_projection_matrices(mtx1, mtx2, R, T)

    os.makedirs("camera_parameters", exist_ok=True)
    save_extrinsics("camera_parameters/rot_trans_c0.dat", np.eye(3), np.zeros((3, 1)))
    save_extrinsics("camera_parameters/rot_trans_c1.dat", R, T)

    return P1, P2, R, T

if __name__ == "__main__":
    stereo_pipeline()
