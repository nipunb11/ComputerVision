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
- **Buddha Subedi**  (`subed042`)
- **Yassin Ali** — (`ali00740`) [GitHub Profile](https://github.com/Y-Elsayed)  
- **Thomas Yip** (`yip00023`) 

University of Minnesota  
CSCI 5561 — Computer Vision

---

## 🏁 Future Work


- **Automatic Whiteboard Detection**  
  Replace the current manual corner selection with automated detection using segmentation models (e.g., Segment Anything Model) or geometric edge-aware methods to improve usability and reduce setup time.

- **Lighting-Consistent Inpainting**  
  Integrate photometric normalization or illumination-aware inpainting models to correct lighting mismatches between original frames and background patches during professor removal.

- **Real-Time Pose Filtering and Joint Suppression**  
  Implement confidence-based joint filtering and Kalman filtering for smoother, more stable 3D reconstructions—reducing hallucinated or jittery movements, especially under occlusion.

- **Pre-Processing of 3D Keypoints for Blender**  
  Filter and smooth noisy keypoints before feeding them into Blender to enhance animation realism and reduce bone instability.

- **Configurable Rendering Parameters**  
  Expose Blender render settings (e.g., resolution, samples, background path) to users via a config file or interface for easier tuning and reproducibility.

- **Lecture Note Generation via AI**  
  Apply OCR and AI-driven summarization on infilled whiteboard frames to auto-generate structured lecture notes—creating an end-to-end educational enhancement tool.

- **Calibration Quality Checks**  
  Add automatic validation tools to assess stereo calibration coverage and detect weaknesses before proceeding to reconstruction.

- **Light Source Estimation for Realism**  
  Explore deep learning-based methods to estimate lighting direction and intensity for better avatar integration into the classroom scene with realistic shadows and highlights.

- **Multi-View Fusion for Pose Accuracy**  
  Incorporate depth estimation and advanced fusion methods to handle occlusions and improve the overall spatial accuracy of the avatar.
