import cv2 as cv
import glob
import numpy as np
import os

def calibrate_camera(images_glob_pattern, save_prefix, world_scaling=0.025, checkerboard_size=(7, 10)):
    """
    Calibrate a camera using checkerboard images and save the intrinsics and distortion coefficients.

    Args:
        images_glob_pattern (str): Glob pattern to find calibration images (e.g., 'left_calib/*.jpg').
        save_prefix (str): Prefix for saving the calibration output files.
        world_scaling (float): Real-world size of a square on the checkerboard in meters (default: 0.025).
        checkerboard_size (tuple): Size of the checkerboard (rows, columns).

    Returns:
        mtx (np.ndarray): Camera intrinsic matrix.
        dist (np.ndarray): Distortion coefficients.
    """
    image_paths = sorted(glob.glob(images_glob_pattern))
    images = [cv.imread(path, 1) for path in image_paths]

    if not images:
        raise FileNotFoundError(f"No images found matching pattern: {images_glob_pattern}")


    rows, columns = checkerboard_size
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2) * world_scaling

    imgpoints, objpoints = [], []

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    if not imgpoints:
        raise ValueError(f"No checkerboard corners were detected in any image for {save_prefix}")

    image_size = (images[0].shape[1], images[0].shape[0])
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    print(f"\n{save_prefix} Camera RMSE: {ret:.4f}")
    print(f"{save_prefix} Camera Matrix:\n{mtx}")
    print(f"{save_prefix} Distortion Coefficients:\n{dist.ravel()}")

    # Save parameters
    os.makedirs("camera_parameters", exist_ok=True)
    os.makedirs("CalibrationData", exist_ok=True)

    with open(os.path.join("camera_parameters", f"{save_prefix}.dat"), "w") as f:
        f.write("intrinsics:\n")
        for row in mtx:
            f.write(" ".join(f"{val:.8e}" for val in row) + "\n")
        f.write("distortion:\n")
        f.write(" ".join(f"{val:.8e}" for val in dist.ravel()) + "\n")

    np.savez(f"CalibrationData/{save_prefix}_calib_data.npz", mtx=mtx, dist=dist)

    return mtx, dist


if __name__ == "__main__":
    # Example usage when run directly
    calibrate_camera('left_calib/*.jpg', 'c0')
    calibrate_camera('right_calib/*.jpg', 'c1')
