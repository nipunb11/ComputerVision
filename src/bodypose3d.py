import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [720, 1280]

# Keypoints to track
pose_keypoints = [0, 16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

def project_3d_to_2d(points3d, P):
    """Project 3D points to 2D using the projection matrix P"""
    points2d = []
    for point in points3d:
        if point[0] == -1:  # Skip invalid points
            points2d.append([-1, -1])
            continue
        point_hom = np.append(point, 1)  # Convert to homogeneous coordinates
        point_proj = P @ point_hom  # Project using P
        point_proj = point_proj / point_proj[2]  # Normalize
        points2d.append([point_proj[0], point_proj[1]])
    return np.array(points2d)

def run_mp(input_stream1, input_stream2, P0, P1):
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    fps = cap0.get(cv.CAP_PROP_FPS)
    print(fps)
    

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('Outputs/output_with_3d_projection.mp4', fourcc, fps, (frame_shape[1], frame_shape[0]))

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        frame0_keypoints = []
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints:
                    continue
                pxl_x = int(round(landmark.x * frame0.shape[1]))
                pxl_y = int(round(landmark.y * frame0.shape[0]))
                cv.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                frame0_keypoints.append([pxl_x, pxl_y])
        else:
            frame0_keypoints = [[-1, -1]] * len(pose_keypoints)

        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                if i not in pose_keypoints:
                    continue
                pxl_x = int(round(landmark.x * frame1.shape[1]))
                pxl_y = int(round(landmark.y * frame1.shape[0]))
                cv.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                frame1_keypoints.append([pxl_x, pxl_y])
        else:
            frame1_keypoints = [[-1, -1]] * len(pose_keypoints)

        kpts_cam1.append(frame1_keypoints)

        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((13, 3))
        kpts_3d.append(frame_p3ds)

        projected_points = project_3d_to_2d(frame_p3ds, P0)

        connections = [
            (1, 3), (1, 2), (2, 8), (1, 7), (2, 4),
            (3, 5), (4, 6), (7, 8), (7, 9), (8, 10),
            (9, 11), (10, 12), (0, 1), (0, 2)
        ]

        for i, point in enumerate(projected_points):
            if point[0] == -1:
                continue
            x, y = int(round(point[0])), int(round(point[1]))
            cv.circle(frame0, (x, y), 5, (0, 255, 0), -1)
            cv.putText(frame0, str(i), (x + 10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for start, end in connections:
            if (projected_points[start][0] == -1 or projected_points[end][0] == -1):
                continue
            start_point = (int(round(projected_points[start][0])), int(round(projected_points[start][1])))
            end_point = (int(round(projected_points[end][0])), int(round(projected_points[end][1])))
            cv.line(frame0, start_point, end_point, (0, 255, 0), 2)

        out.write(frame0)

        display_scale = 0.5
        frame0_display = cv.resize(frame0, (0, 0), fx=display_scale, fy=display_scale)
        frame1_display = cv.resize(frame1, (0, 0), fx=display_scale, fy=display_scale)

        cv.imshow('Left Camera with 3D Projection', frame0_display)
        cv.imshow('Right Camera', frame1_display)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
    out.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':
    input_stream1 = 'media/oleft_pose.mp4'
    input_stream2 = 'media/oright_pose.mp4'

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    write_keypoints_to_disk('3dpart/kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('3dpart/kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('3dpart/kpts_3d.dat', kpts_3d)

    cap_out = cv.VideoCapture('output_with_3d_projection.mp4')
    while cap_out.isOpened():
        ret, frame = cap_out.read()
        if not ret:
            break
        cv.imshow('Final 3D Projection on Left Camera', frame)
        if cv.waitKey(30) & 0xFF == 27:
            break
    cap_out.release()
    cv.destroyAllWindows()
