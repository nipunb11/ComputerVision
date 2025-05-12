# video_splitter.py

from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def split_dual_videos(
    left_input: str,
    right_input: str,
    split_time: int,
    left_calib_output: str,
    left_pose_output: str,
    right_calib_output: str,
    right_pose_output: str
):
    """
    Splits two synchronized videos (left and right) into calibration and pose segments.

    Args:
        left_input (str): Path to the left input video file.
        right_input (str): Path to the right input video file.
        split_time (int): Time in seconds to split both videos.
        left_calib_output (str): Output path for the left calibration segment.
        left_pose_output (str): Output path for the left pose segment.
        right_calib_output (str): Output path for the right calibration segment.
        right_pose_output (str): Output path for the right pose segment.
    """
    # Get durations
    left_duration = VideoFileClip(left_input).duration
    right_duration = VideoFileClip(right_input).duration

    # Cut left video
    ffmpeg_extract_subclip(left_input, 0, split_time, targetname=left_calib_output)
    ffmpeg_extract_subclip(left_input, split_time, left_duration, targetname=left_pose_output)

    # Cut right video
    ffmpeg_extract_subclip(right_input, 0, split_time, targetname=right_calib_output)
    ffmpeg_extract_subclip(right_input, split_time, right_duration, targetname=right_pose_output)
