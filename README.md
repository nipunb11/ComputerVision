# Enhancing Classroom Comprehension with Multi-View 3D Reconstruction and Whiteboard Transformation

This project presents an end-to-end computer vision system for improving the visibility and accessibility of classroom lectures. By combining multi-view 3D reconstruction, semantic inpainting, and virtual avatar animation, we eliminate instructor occlusion, preserve whiteboard content, and generate a smartboard-style digital output. The system also includes a frontal-view whiteboard transformation and compositing pipeline for enhanced readability.

Developed as part of the **CSCI 5561: Computer Vision** course at the University of Minnesota.

---

## 📽️ What This Project Does

- Calibrates a dual-camera setup using a checkerboard pattern.
- Detects 2D pose keypoints of an instructor using MediaPipe, then reconstructs a 3D pose using DLT triangulation.
- Removes the instructor from the scene using pose-guided inpainting.
- Animates a rigged digital avatar using Blender based on reconstructed joint positions.
- Applies a homography transformation to present the whiteboard in a front-facing “smartboard” format.
- Composites the avatar and restored whiteboard into a final output video.

---

## 🧑‍🏫 Before You Start

Before running the system:

1. **Checkerboard Calibration**
   - Use a **7x10 checkerboard pattern**.
   - Ensure **full overlap** between the two camera views.
   - Move the board to **multiple angles and positions** to cover the entire frame for both cameras.

2. **Prepare Input Videos**
   - You’ll need two input videos: one from the left camera, one from the right.
   - Upload both videos to the `media/` directory.

---

## ▶️ How to Run

1. Open `main.py`.
2. Set the following parameters:
   - `left_video`: path to left camera video (e.g., `media/left.mp4`)
   - `right_video`: path to right camera video (e.g., `media/right.mp4`)
   - `timestamp`: the time (in seconds) where calibration ends and the avatar animation should begin.
   - `resolution_percentage`: scale factor (e.g., 0.5 for 50%) to speed up processing.

3. Run the script.

---

## 📂 Output

All generated outputs will be saved to the `outputs/` directory:

- `final_result.mp4`: The fully composited lecture video with avatar and inpainted whiteboard.
- `smartboard_output.mp4`: The transformed front-facing whiteboard view (without the instructor).

---


## 📦 Requirements

Install dependencies using `pip install -r requirements.txt`, or individually:

```bash
bpy==4.4.0  
mathutils==3.3.0  
matplotlib==3.7.2  
mediapipe==0.8.7  
moviepy==1.0.3  
numpy==2.2.5  
opencv_contrib_python==4.5.1.48  
opencv_python_headless==4.10.0.84  
scipy==1.15.3  
```
> ⚠️ **Note:** `bpy` (Blender Python API) may require a Blender-compatible Python environment.  
> Refer to [Blender’s official scripting documentation](https://docs.blender.org/api/current/info_quickstart.html) for setup instructions.

## 📌 Notes
- Performance may vary depending on CPU/GPU specs. The current implementation is optimized for offline processing.
- Make sure to use the same camera models/settings for accurate stereo calibration.
- Output quality is highly dependent on the visibility of calibration patterns and pose estimation fidelity.
  
---

## 👥 Team

- **Nipun Bhatnagar** (`bhatn058`)  
- **Buddha Subedi**  
- **Yassin Ali** — [GitHub Profile](https://github.com/Y-Elsayed)  
- **Thomas Yip**

University of Minnesota  
CSCI 5561 — Computer Vision

---

## 🏁 Future Work

- Real-time pose filtering and smoothing.
- Automatic whiteboard detection (eliminating manual corner selection).
- Integration with OCR and lecture summarization tools.

