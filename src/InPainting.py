import cv2
import numpy as np
import mediapipe as mp

def inpaint_person(input_path, output_path="inpainted.mp4", buffer_size=50, detection_scale=0.5, expand=100):
    """
    Inpaints a person from a video using pose detection and background modeling.
    
    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        buffer_size (int): Number of frames used for background modeling.
        detection_scale (float): Downscale factor for pose detection (0–1).
        expand (int): Number of pixels to expand the bounding box around the detected person.
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
        
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    # Frame buffer for background modeling
    frame_buffer = []
    cached_bg = None

    def get_most_representative_frame(buffer):
        n = len(buffer)
        resized = [cv2.resize(f, (160, 90)) for f in buffer]
        diffs = np.zeros(n)

        for i in range(n):
            total_diff = 0
            for j in range(n):
                if i != j:
                    diff = cv2.absdiff(resized[i], resized[j])
                    total_diff += np.mean(diff)
            diffs[i] = total_diff

        best_index = int(np.argmin(diffs))
        return buffer[best_index]

    # Main loop
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        print(f"Processing frame {current_frame}/{total_frames}...")

        # Downscale for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
        frame_rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb_small)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            x_min = int(min(x_coords) * original_width)
            x_max = int(max(x_coords) * original_width)
            y_min = int(min(y_coords) * original_height)
            y_max = int(max(y_coords) * original_height)

            x_min = max(0, x_min - expand)
            x_max = min(original_width, x_max + expand)
            y_min = max(0, y_min - expand)
            y_max = min(original_height, y_max + expand)

            # Refresh background every 20 frames
            if len(frame_buffer) >= buffer_size and current_frame % 20 == 0:
                cached_bg = get_most_representative_frame(frame_buffer)

            # Inpaint region
            if cached_bg is not None:
                frame[y_min:y_max, x_min:x_max] = cached_bg[y_min:y_max, x_min:x_max]

        # Update background buffer
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        out.write(frame)

    cap.release()
    out.release()
    pose.close()
    cv2.destroyAllWindows()
    print(f"Inpainting complete. Output saved to: {output_path}")
