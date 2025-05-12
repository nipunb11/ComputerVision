import cv2
import numpy as np
import os

def select_points_from_video(video_path, scale_factor=1.0):
    """
    Opens the first frame of a video, resizes it by scale_factor, 
    and lets the user select 4 points by clicking on them.

    Args:
        video_path (str): Path to the video file.
        scale_factor (float): Factor by which to scale the frame (e.g., 0.5 for 50%).

    Returns:
        list: List of 4 (x, y) tuples of selected points on the scaled frame.
    """
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, str(len(points)), (x+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('First Frame', frame_copy)
            if len(points) == 4:
                print("\nSelected points:")
                for i, (px, py) in enumerate(points):
                    print(f"Point {i+1}: ({px}, {py})")
                # Close the window after 4 points are selected
                cv2.destroyWindow('First Frame')

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read the video from path: {video_path}")

    # Resize the frame
    new_width = int(frame.shape[1] * scale_factor)
    new_height = int(frame.shape[0] * scale_factor)
    frame = cv2.resize(frame, (new_width, new_height))

    global frame_copy
    frame_copy = frame.copy()

    cv2.namedWindow('First Frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('First Frame', click_event)
    cv2.resizeWindow('First Frame', frame.shape[1], frame.shape[0])

    print("\nClick to select 4 points. Press 'r' to reset, 'q' to quit.")
    cv2.imshow('First Frame', frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) == 4:  # Exit if 'q' pressed or 4 points selected
            break
        elif key == ord('r'):
            points = []
            frame_copy = frame.copy()
            cv2.imshow('First Frame', frame_copy)

    cv2.destroyAllWindows()
    return points



def warp_whiteboard_video(input_video_path, pts, output_video_path="Outputs/smartboard_output.mp4", output_dir="Outputs"):
    """
    Warps perspective of video using pre-selected 4 points so whiteboard appears front-facing.
    pts: List of 4 points in order [TL, TR, BR, BL] as (x,y) tuples
    """
    
    os.makedirs(output_dir, exist_ok=True)

    def compute_output_size(pts):
        tl, tr, br, bl = pts
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        height = int(max(height_left, height_right))

        return (width, height)

    # Validate points
    if len(pts) != 4:
        raise ValueError("Exactly 4 points must be provided")
    
    # Convert to numpy array
    src_pts = np.array(pts, dtype=np.float32)
    
    # Compute output size and destination points
    output_size = compute_output_size(src_pts)
    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)

    # Get perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Load video
    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), f"Failed to open input video at {input_video_path}"
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, output_size)

    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        warped = cv2.warpPerspective(frame, M, output_size)
        writer.write(warped)

    cap.release()
    writer.release()
    print(f"Smartboard video saved at: {output_video_path}")