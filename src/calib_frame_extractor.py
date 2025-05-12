import cv2
import numpy as np
import os
import shutil

def extract_checkerboard_frames(
    left_video_path: str,
    right_video_path: str,
    left_save_dir: str = "left_calib",
    right_save_dir: str = "right_calib",
    checkerboard_size: tuple = (7, 10),
    interval_sec: int = 2
):
    """
    Extract frames containing checkerboards at regular time intervals from two synchronized videos.

    Args:
        left_video_path (str): Path to the left video file.
        right_video_path (str): Path to the right video file.
        left_save_dir (str): Directory to save detected frames from the left video.
        right_save_dir (str): Directory to save detected frames from the right video.
        checkerboard_size (tuple): Size of the checkerboard (rows, cols).
        interval_sec (int): Interval (in seconds) between frame samples.
    """

    def detect_checkerboard(frame, checkerboard_size):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        return ret, corners if ret else None

    def setup_folder(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Prepare directories
    setup_folder(left_save_dir)
    setup_folder(right_save_dir)

    # Load videos
    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)

    fps_left = cap_left.get(cv2.CAP_PROP_FPS)
    fps_right = cap_right.get(cv2.CAP_PROP_FPS)

    total_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = min(total_frames_left / fps_left, total_frames_right / fps_right)

    print(f"Processing every {interval_sec}s up to {total_time:.2f}s")

    t = 0
    while True:
        frame_idx_left = int(t * fps_left)
        frame_idx_right = int(t * fps_right)

        if frame_idx_left >= total_frames_left or frame_idx_right >= total_frames_right:
            break

        cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_left)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_right)

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            break

        found_left, _ = detect_checkerboard(frame_left, checkerboard_size)
        found_right, _ = detect_checkerboard(frame_right, checkerboard_size)
        print(found_left, found_right)

        if found_left and found_right:
            timestamp = int(t)
            cv2.imwrite(os.path.join(left_save_dir, f"left_calib_{timestamp}s.jpg"), frame_left)
            cv2.imwrite(os.path.join(right_save_dir, f"right_calib_{timestamp}s.jpg"), frame_right)
            print(f"Saved checkerboard at {timestamp}s")

        t += interval_sec

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
