from video_splitter import split_dual_videos
from calib_frame_extractor import extract_checkerboard_frames
from camera_calibrator import calibrate_camera
from stereo_triangulation import stereo_pipeline
from bodypose3d import run_mp
from utils import get_projection_matrix, write_keypoints_to_disk
from InPainting import inpaint_person
from VideoCompositor import process_videos
from Whiteboard import warp_whiteboard_video, select_points_from_video

import subprocess


# Define base names
left_video = "oleft"
right_video = "oright"
resolution_num = 80
timestamp = "00:01:52"




def hms_to_seconds(hms):
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s

split_time_seconds = hms_to_seconds(timestamp)
print(split_time_seconds)  # Output: 6360


# Step 1: Split videos
split_dual_videos(
    left_input=f"media/{left_video}.mp4",
    right_input=f"media/{right_video}.mp4",
    split_time=split_time_seconds,
    left_calib_output=f"media/{left_video}_calib.mp4",
    left_pose_output=f"media/{left_video}_pose.mp4",
    right_calib_output=f"media/{right_video}_calib.mp4",
    right_pose_output=f"media/{right_video}_pose.mp4"
)

scale_factor = resolution_num/100
selected = select_points_from_video(f"media/{left_video}_pose.mp4", scale_factor=scale_factor)




# Step 2: Extract calibration frames
extract_checkerboard_frames(
    left_video_path=f"media/{left_video}_calib.mp4",
    right_video_path=f"media/{right_video}_calib.mp4",
    left_save_dir="left_calib",
    right_save_dir="right_calib",
    checkerboard_size=(7, 10),
    interval_sec=2
)

# Step 3: Calibrate each camera
mtx0, dist0 = calibrate_camera('left_calib/*.jpg', 'c0')
mtx1, dist1 = calibrate_camera('right_calib/*.jpg', 'c1')

# Step 4: Stereo pipeline
P1, P2, R, T = stereo_pipeline()

# Step 5: Run 3D pose estimation
input_stream1 = f'media/{left_video}_pose.mp4'
input_stream2 = f'media/{right_video}_pose.mp4'

P0 = get_projection_matrix(0)
P1 = get_projection_matrix(1)

kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

# Step 6: Save keypoints
write_keypoints_to_disk('Outputs/kpts_cam0.dat', kpts_cam0)
write_keypoints_to_disk('Outputs/kpts_cam1.dat', kpts_cam1)
write_keypoints_to_disk('Outputs/kpts_3d.dat', kpts_3d)


inpaint_person(f'media/{left_video}_pose.mp4', output_path="Outputs/inpainted.mp4")

resolution = str(resolution_num)

subprocess.run([
    "blender", "-b", "Miles.blend",   # PICK BETWEEN MILES.BLEND and RIGGEDMODEL.BLEND
    "--python", "src/blender.py",
    "--",
    "--keypoints", "Outputs/kpts_3d.dat",
    "--video", "Outputs/inpainted.mp4",
    "--output", "Outputs/render.mp4",
    "--resolution", resolution,
    "--render"
])


process_videos(
    mask_source_path="Outputs/render.mp4",
    pose_video_path=f'media/{left_video}_pose.mp4',
    final_background_path="Outputs/inpainted.mp4",
    result_path="Outputs/final_result.mp4"
)


warp_whiteboard_video(f'outputs/final_result.mp4', selected)
