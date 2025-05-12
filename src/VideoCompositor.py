import cv2
import mediapipe as mp
import numpy as np

def process_videos(mask_source_path, pose_video_path, final_background_path="output.mp4", result_path="final_result.mp4"):
    """
    Overlays person region from mask_source_path onto final_background_path using segmentation mask.
    Pose detection is done on pose_video_path to track movement and define ROI for segmentation.
    
    Args:
        mask_source_path (str): Path to video containing person to overlay.
        pose_video_path (str): Path to video for pose tracking (guides mask position).
        final_background_path (str): Path to background video.
        result_path (str): Path to save final composited video.
    """
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Open videos
    pose_cap = cv2.VideoCapture(pose_video_path)
    mask_cap = cv2.VideoCapture(mask_source_path)
    background_cap = cv2.VideoCapture(final_background_path)

    width = int(pose_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    height = int(pose_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
    fps = pose_cap.get(cv2.CAP_PROP_FPS)

    total_frames = min(
        int(pose_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    # Padding for ROI
    padding_up, padding_left, padding_right, padding_down = 80, 120, 100, 30
    prev_mask = None
    process_interval = 5
    last_processed_frame = -process_interval

    for frame_idx in range(total_frames):
        ret_pose, pose_frame = pose_cap.read()
        ret_mask, mask_source_frame = mask_cap.read()
        ret_bg, background_frame = background_cap.read()

        if not ret_pose or not ret_mask or not ret_bg:
            break

        pose_frame = cv2.resize(pose_frame, (width, height))
        mask_source_frame = cv2.resize(mask_source_frame, (width, height))
        background_frame = cv2.resize(background_frame, (width, height))

        if frame_idx - last_processed_frame >= process_interval:
            pose_rgb = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(pose_rgb)

            if pose_results.pose_landmarks:
                visible_coords = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in pose_results.pose_landmarks.landmark
                    if lm.visibility > 0.5
                ]

                if visible_coords:
                    pts = np.array(visible_coords)
                    x_min = max(0, np.min(pts[:, 0]) - padding_left)
                    x_max = min(width, np.max(pts[:, 0]) + padding_right)
                    y_min = max(0, np.min(pts[:, 1]) - padding_up)
                    y_max = min(height, np.max(pts[:, 1]) + padding_down)

                    roi = mask_source_frame[y_min:y_max, x_min:x_max]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    seg_results = segmentation.process(roi_rgb)
                    seg_mask_roi = (seg_results.segmentation_mask > 0.3).astype(np.uint8) * 255

                    seg_mask_full = np.zeros((height, width), dtype=np.uint8)
                    seg_mask_full[y_min:y_max, x_min:x_max] = seg_mask_roi

                    dilate_kernel = np.ones((9, 9), np.uint8)
                    seg_mask_full = cv2.dilate(seg_mask_full, dilate_kernel, iterations=2)

                    morph_kernel = np.ones((5, 5), np.uint8)
                    seg_mask_full = cv2.morphologyEx(seg_mask_full, cv2.MORPH_CLOSE, morph_kernel)

                    if prev_mask is not None:
                        seg_mask_full = cv2.addWeighted(seg_mask_full, 0.7, prev_mask, 0.3, 0)

                    prev_mask = seg_mask_full
                    last_processed_frame = frame_idx

        if prev_mask is not None:
            blended_mask = cv2.GaussianBlur(prev_mask, (25, 25), 0).astype(float) / 255.0
            blended_mask = np.stack([blended_mask] * 3, axis=2)

            person = mask_source_frame.astype(float)
            bg = background_frame.astype(float)

            result = person * blended_mask + bg * (1 - blended_mask)
            result = result.astype(np.uint8)
        else:
            result = background_frame

        out.write(result)
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")

    pose_cap.release()
    mask_cap.release()
    background_cap.release()
    out.release()
    print(f"Final compositing complete with segmentation. Output saved to {result_path}")
