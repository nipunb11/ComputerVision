#!/usr/bin/env python3
"""
Blender Motion Capture Animation Script
--------------------------------------
This script imports 3D keypoints data and applies it to an armature in Blender.
It creates a character animation from motion capture data and renders it with
a video background.
"""

import bpy
import numpy as np
import os
import math
import mathutils
import sys
import argparse


# =============================================
# DATA LOADING FUNCTIONS
# =============================================

def read_keypoints(filename, selected_indices=None):
    """
    Read 3D keypoints data from file and format it for Blender.
    
    Args:
        filename: Path to keypoints file
        selected_indices: List of keypoint indices to use (default: indices 0-12)
        
    Returns:
        Numpy array of formatted keypoints
    """
    if selected_indices is None:
        selected_indices = list(range(0, 13))
        
    with open(filename, 'r') as fin:
        kpts = []
        for line in fin:
            if line.strip() == '':
                break
            line = list(map(float, line.split()))
            frame = []
            for idx in selected_indices:
                frame.append(line[idx*3 : idx*3+3])
            kpts.append(np.array(frame))
    
    # Format: swap Y and Z axes, flip Z
    kpts = np.array(kpts)
    kpts = kpts[:, :, [0, 2, 1]]  # Swap Y and Z
    kpts[:, :, 2] *= -1           # Flip Z
    
    return kpts


# =============================================
# SCENE SETUP FUNCTIONS
# =============================================

def setup_scene(fps=30):
    """
    Configure basic scene parameters.
    
    Args:
        fps: Frames per second for animation
    """
    bpy.context.scene.render.fps = fps
    
    # Clear any existing handlers for frame change
    handlers = bpy.app.handlers.frame_change_pre
    handlers[:] = [h for h in handlers if h.__name__ != "update_bone_rotation"]


def create_tracking_objects(p3ds, joint_radius=0.05):
    """
    Create sphere objects for each joint to track.
    
    Args:
        p3ds: Keypoints data
        joint_radius: Radius of the tracking spheres
        
    Returns:
        Dictionary of joint objects
    """
    joints = {}
    for i in range(p3ds.shape[1]):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=joint_radius)
        sphere = bpy.context.active_object
        sphere.name = f"Joint_{i}"
        joints[i] = sphere
        
    return joints


def setup_visibility(keep_visible=None):
    if keep_visible is None:
        keep_visible = "Cube"
        
    keep_visible_names = {keep_visible, "Sun"}  # <- Also keep Sun visible
    
    for obj in bpy.data.objects:
        if obj.name not in keep_visible_names:
            obj.hide_render = True
            
    print(f"All objects except {keep_visible_names} are now hidden in render (still visible in viewport).")



# =============================================
# ANIMATION FUNCTIONS
# =============================================

def animate_joints(p3ds, joints, frame_spacing=1):
    """
    Create keyframes for joint positions.
    
    Args:
        p3ds: Keypoints data
        joints: Dictionary of joint objects
        frame_spacing: Number of frames between keyframes
        
    Returns:
        Total number of frames
    """
    cube_obj = bpy.data.objects.get("Cube")
    if not cube_obj:
        print("Warning: Cube object not found for visibility toggling.")
    
    current_frame = 1
    for frame_idx, frame_pts in enumerate(p3ds):
        # Check if this frame has valid points
        frame_has_valid_data = not np.any(frame_pts[0] == -1)
        
        # Set cube visibility keyframe if cube exists
        if cube_obj:
            cube_obj.hide_viewport = not frame_has_valid_data
            cube_obj.hide_render = not frame_has_valid_data
            cube_obj.keyframe_insert(data_path="hide_viewport", frame=current_frame)
            cube_obj.keyframe_insert(data_path="hide_render", frame=current_frame)
        
        # Animate joint positions when they're valid
        for i, pos in enumerate(frame_pts):
            if not np.any(pos == -1):  # Only animate valid points
                joints[i].location = pos
                joints[i].keyframe_insert(data_path="location", frame=current_frame)
        
        current_frame += frame_spacing
    
    # Scene settings
    bpy.context.scene.frame_end = current_frame
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_set(1)
    
    print(f"Animated keypoints. Total frames: {current_frame}")
    return current_frame


# =============================================
# RIGGING FUNCTIONS
# =============================================

def find_valid_frame(p3ds, required_joints):
    """
    Find the first frame with valid data for specified joints.
    
    Args:
        p3ds: Keypoints data
        required_joints: List of joint indices that must be valid
        
    Returns:
        Frame index and frame data
    """
    for frame_idx, frame_pts in enumerate(p3ds):
        valid = True
        for joint_idx in required_joints:
            if np.any(frame_pts[joint_idx] == -1):
                valid = False
                break
                
        if valid:
            return frame_idx, frame_pts
            
    raise Exception(f"No frame with valid points for joints {required_joints} found!")


def scale_armature(p3ds, armature_name="metarig", shoulder_bones=("shoulder.L", "shoulder.R")):
    """
    Scale armature to match motion capture data based on shoulder width.
    
    Args:
        p3ds: Keypoints data
        armature_name: Name of the armature object
        shoulder_bones: Tuple of bone names for left and right shoulders
        
    Returns:
        Scale factor applied
    """
    armature_obj = bpy.data.objects.get(armature_name)
    if not armature_obj:
        print(f"Warning: Armature '{armature_name}' not found")
        return 1.0
    
    # Find valid frame for shoulder measurements
    valid_frame_idx, valid_frame = find_valid_frame(p3ds, [1, 2])
    print(f"Using frame {valid_frame_idx} for shoulder width calculation")
    
    # Compute distance between animated shoulders
    shoulder_L_pos = valid_frame[1]  # Left shoulder
    shoulder_R_pos = valid_frame[2]  # Right shoulder
    
    shoulder_vec = np.array(shoulder_L_pos) - np.array(shoulder_R_pos)
    animated_width = np.linalg.norm(shoulder_vec)
    print(f"Animated shoulder width: {animated_width}")
    
    # Compute current shoulder width in rig
    shoulder_L_pos = armature_obj.data.bones[shoulder_bones[0]].tail_local
    shoulder_R_pos = armature_obj.data.bones[shoulder_bones[1]].tail_local
    
    rig_vec = np.array(shoulder_L_pos) - np.array(shoulder_R_pos)
    rig_width = np.linalg.norm(rig_vec)
    print(f"Rig shoulder width: {rig_width}")
    
    # Compute scale factor (3/4 is a tuning factor)
    scale_factor = (animated_width / rig_width) * 6/10
    armature_obj.scale = [s * scale_factor for s in armature_obj.scale]
    print(f"Scaled armature by factor: {scale_factor:.4f} to match shoulder width.")
    
    return scale_factor


def setup_armature_tracking(armature_name="metarig", mesh_name="Cube"):
    """
    Set up parent-child relationship between armature and mesh.
    
    Args:
        armature_name: Name of the armature object
        mesh_name: Name of the mesh object
    """
    armature_obj = bpy.data.objects.get(armature_name)
    mesh_obj = bpy.data.objects.get(mesh_name)
    
    if not armature_obj:
        print(f"Warning: Armature '{armature_name}' not found")
        return
    
    if not mesh_obj:
        print(f"Warning: Mesh '{mesh_name}' not found")
        return
    
    # Create parent relationship
    mesh_obj.parent = armature_obj
    print(f"Set {mesh_name} as child of {armature_name}")


def setup_spine_tracking(p3ds, armature_name="metarig", spine_bone="spine", target_joint=0):
    """
    Set up spine bone tracking to target joint.
    
    Args:
        p3ds: Keypoints data
        armature_name: Name of the armature object
        spine_bone: Name of the spine bone
        target_joint: Joint index to track
    """
    armature_obj = bpy.data.objects.get(armature_name)
    target_obj = bpy.data.objects.get(f"Joint_{target_joint}")
    
    if not armature_obj:
        print(f"Warning: Armature '{armature_name}' not found")
        return
        
    if not target_obj:
        print(f"Warning: Target joint 'Joint_{target_joint}' not found")
        return
    
    # Switch to Pose Mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    # Add a Copy Location constraint to the 'spine' bone
    origin_bone = armature_obj.pose.bones[spine_bone]
    constraint = origin_bone.constraints.new(type='COPY_LOCATION')
    constraint.name = "SpineCopyLocation"
    constraint.target = target_obj
    constraint.use_offset = True
    constraint.owner_space = 'WORLD'
    constraint.target_space = 'WORLD'
    
    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Set {spine_bone} bone to track Joint_{target_joint}")
    
    # Find a frame with valid spine and hip joints
    try:
        valid_height_frame_idx, _ = find_valid_frame(p3ds, [0, 7, 8])
        print(f"Using frame {valid_height_frame_idx} for height calculation")
        
        spine_pos = p3ds[valid_height_frame_idx][0]
        hip_left_pos = p3ds[valid_height_frame_idx][7]
        hip_right_pos = p3ds[valid_height_frame_idx][8]
        
        # Get the "spine" bone
        spine_bone_obj = armature_obj.pose.bones.get(spine_bone)
        
        if spine_bone_obj:
            # World location of spine bone
            bone_world_matrix = armature_obj.matrix_world @ spine_bone_obj.matrix
            bone_world_location = bone_world_matrix.translation
            bone_z = bone_world_location.z
            
            # Average Z of hip joints
            hips_avg_z = (hip_left_pos[2] + hip_right_pos[2]) / 2.0
            
            # Difference
            z_difference = bone_z - hips_avg_z
            
            print(f"Z difference (spine - hips avg): {z_difference:.4f}")
            armature_obj.location.z -= z_difference
        else:
            print(f"Spine bone '{spine_bone}' not found!")
    except Exception as e:
        print(f"Error adjusting height: {e}")


def setup_bone_constraints(armature_name="metarig", bone_joint_mapping=None):
    """
    Set up Damped Track constraints for bones to track joints.
    
    Args:
        armature_name: Name of the armature object
        bone_joint_mapping: Dictionary mapping bone names to joint indices
    """
    if bone_joint_mapping is None:
        bone_joint_mapping = {
            "spine": 0,
            "upper_arm.R": 4,
            "upper_arm.L": 3,
            "forearm.L": 5,
            "forearm.R": 6,
            "thigh.R": 10,
            "thigh.L": 9,
            "shin.R": 12,
            "shin.L": 11
        }
    
    armature_obj = bpy.data.objects.get(armature_name)
    
    if not armature_obj:
        print(f"Warning: Armature '{armature_name}' not found")
        return
    
    # Convert joint indices to object names
    bone_to_object_mapping = {bone: f"Joint_{joint_idx}" for bone, joint_idx in bone_joint_mapping.items()}
    
    # Activate armature and switch to Pose Mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    # Apply Damped Track constraints
    for bone_name, joint_name in bone_to_object_mapping.items():
        target = bpy.data.objects.get(joint_name)
        if target:
            bone = armature_obj.pose.bones.get(bone_name)
            if bone:
                constraint = bone.constraints.new(type='DAMPED_TRACK')
                constraint.name = f"DampedTrack_{joint_name}"
                constraint.target = target
                constraint.track_axis = 'TRACK_Y'  # or 'TRACK_Z' depending on bone orientation
                print(f"Applied Damped Track from {bone_name} to {joint_name}")
            else:
                print(f"Bone {bone_name} not found.")
        else:
            print(f"Target joint {joint_name} not found.")
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')


# =============================================
# SPINE ROTATION SETUP
# =============================================

def create_line_object(joint1, joint2, line_name="Line_Object"):
    """
    Create a line object (cylinder) between two joints.
    
    Args:
        joint1: First joint object
        joint2: Second joint object
        line_name: Name for the line object
        
    Returns:
        Line object
    """
    # Create a cylinder with a very small radius
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.05, depth=1, location=(0, 0, 0))
    line_object = bpy.context.active_object
    line_object.name = line_name
    
    # Set the cylinder rotation to align with the direction from joint1 to joint2
    direction = joint2.location - joint1.location
    distance = direction.length
    
    # Align the cylinder along the direction
    line_object.rotation_mode = 'QUATERNION'
    line_object.rotation_quaternion = direction.to_track_quat('Z', 'Y')  # Align along Z axis
    line_object.scale = (1, 1, distance)  # Scale based on distance between the joints
    
    return line_object


def update_bone_rotation(scene):
    """
    Frame change handler to update bone rotation.
    """
    try:
        joint1 = bpy.data.objects["Joint_1"]
        joint2 = bpy.data.objects["Joint_2"]
        armature = bpy.data.objects["metarig"]
        pose_bone = armature.pose.bones["spine"]
        line_object_name = "Line_Object"
        
        # Check if the current frame has valid joints
        joint1_is_valid = not (np.isclose(joint1.location[0], -1) and 
                              np.isclose(joint1.location[1], -1) and 
                              np.isclose(joint1.location[2], -1))
        joint2_is_valid = not (np.isclose(joint2.location[0], -1) and 
                              np.isclose(joint2.location[1], -1) and 
                              np.isclose(joint2.location[2], -1))
        
        # Only proceed if both joints are valid
        if joint1_is_valid and joint2_is_valid:
            # Calculate the vector between the two joints (XY only)
            vec = joint2.location.xy - joint1.location.xy
            
            # Compute the angle in radians (2D)
            angle = math.atan2(vec.y, vec.x)
            
            # Rotate 180 degrees to match orientation if needed
            angle_perpendicular = angle + math.radians(180)
            
            # Apply the rotation only along the Z-axis
            rot_quat = mathutils.Quaternion((0, 1, 0), angle_perpendicular)
            
            # Apply it in local space
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = rot_quat
            
            # Update or create line object
            if line_object_name not in bpy.data.objects:
                # Create the line object when it doesn't exist
                create_line_object(joint1, joint2)
            else:
                # Update the existing line object
                line_object = bpy.data.objects[line_object_name]
                line_object.location = (joint1.location + joint2.location) / 2  # Midpoint between joints
                direction = joint2.location - joint1.location
                line_object.rotation_mode = 'QUATERNION'
                line_object.rotation_quaternion = direction.to_track_quat('Z', 'Y')  # Rotate along Z-axis
                distance = direction.length
                line_object.scale = (1, 1, distance)  # Scale based on distance
    
    except Exception as e:
        print(f"Error updating bone rotation: {e}")


def setup_spine_rotation_handler():
    """
    Set up frame change handler for spine rotation.
    """
    # Ensure previous handlers are removed
    handlers = bpy.app.handlers.frame_change_pre
    handlers[:] = [h for h in handlers if h.__name__ != "update_bone_rotation"]
    
    # Add the frame change handler for bone rotation updates
    update_bone_rotation.__name__ = "update_bone_rotation"  # Name the function
    bpy.app.handlers.frame_change_pre.append(update_bone_rotation)
    print("Added spine rotation handler")


def bake_spine_rotation(joint1_name="Joint_1", joint2_name="Joint_2", armature_name="metarig", bone_name="spine"):
    """
    Bake spine rotation to keyframes for rendering.
    
    Args:
        joint1_name: Name of first joint
        joint2_name: Name of second joint
        armature_name: Name of the armature
        bone_name: Name of the bone to rotate
    """
    armature_obj = bpy.data.objects.get(armature_name)
    
    if not armature_obj:
        print(f"Warning: Armature '{armature_name}' not found")
        return
    
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    pose_bone = armature_obj.pose.bones[bone_name]
    pose_bone.rotation_mode = 'QUATERNION'
    
    for f in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(f)
        
        joint1 = bpy.data.objects[joint1_name]
        joint2 = bpy.data.objects[joint2_name]
        
        joint1_valid = not np.allclose(joint1.location[:], [-1, -1, -1])
        joint2_valid = not np.allclose(joint2.location[:], [-1, -1, -1])
        
        if joint1_valid and joint2_valid:
            vec = joint2.location.xy - joint1.location.xy
            angle = math.atan2(vec.y, vec.x)
            angle_perpendicular = angle + math.radians(180)
            
            rot_quat = mathutils.Quaternion((0, 1, 0), angle_perpendicular)
            pose_bone.rotation_quaternion = rot_quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=f)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Spine bone rotation baked to keyframes for rendering")
    
    # Hide line object
    line_obj = bpy.data.objects.get("Line_Object")
    if line_obj:
        line_obj.hide_render = True
        print("Line_Object hidden from render")


# =============================================
# CAMERA AND RENDERING FUNCTIONS
# =============================================

def setup_camera(camera_name="FacingFront_Camera", focal_length=4.25, sensor_width=4.8):
    """
    Set up a camera for rendering.
    
    Args:
        camera_name: Name for the camera
        focal_length: Focal length in mm
        sensor_width: Sensor width in mm
        
    Returns:
        Camera object
    """
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.active_object
    camera.name = camera_name
    
    # Rotate camera to face positive Y
    camera.rotation_euler = (math.radians(90), 0, 0)  # 90 degrees around X axis
    
    # Set intrinsics
    camera.data.sensor_width = sensor_width  # Sensor width in mm
    camera.data.lens = focal_length         # Focal length in mm
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    return camera


def setup_background_video(camera, video_path):
    """
    Set up background video for the camera.
    
    Args:
        camera: Camera object
        video_path: Path to video file
    """
    camera.data.show_background_images = True
    bg = camera.data.background_images.new()
    
    # Load the video file
    bg.image = bpy.data.images.load(video_path)
    bg.source = 'IMAGE'
    
    # Frame settings
    bg.image_user.frame_start = 1
    bg.image_user.use_auto_refresh = True
    
    print(f"Background video loaded: {video_path}")


def setup_compositing(video_path, total_frames):
    """
    Set up compositing nodes for rendering with video background.
    
    Args:
        video_path: Path to video file
    """
    # Set Film Transparent
    bpy.context.scene.render.film_transparent = True
    
    # Enable Compositing Nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    
    # Clear all existing nodes
    nodes.clear()
    
    # Create required nodes
    render_layers = nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (-300, 300)
    
    alpha_over = nodes.new(type='CompositorNodeAlphaOver')
    alpha_over.location = (100, 300)
    alpha_over.inputs[0].default_value = 1.0
    
    image_node = nodes.new(type='CompositorNodeImage')
    image_node.location = (-600, 0)
    
    # Create a new image datablock manually
    new_image = bpy.data.images.new(name="BackgroundVideo", width=1920, height=1080)
    new_image.source = 'MOVIE'
    new_image.filepath = video_path
    new_image.reload()
    
    # Assign the image to the Image node
    image_node.image = new_image
    
    # Set the image user properties
    image_node.use_auto_refresh = True
    image_node.frame_start = 1
    image_node.frame_duration = total_frames
    
    # Scale Node (fit to render size)
    scale_node = nodes.new(type='CompositorNodeScale')
    scale_node.space = 'RENDER_SIZE'
    scale_node.location = (-300, 0)
    
    # Composite Output Node
    composite_out = nodes.new(type='CompositorNodeComposite')
    composite_out.location = (400, 300)
    
    # Link nodes together
    links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
    links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
    links.new(alpha_over.outputs['Image'], composite_out.inputs['Image'])
    
    print("Compositing setup complete")


def setup_lighting():
    """
    Set up lighting for the scene.
    """
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    
    # Link light object to the current collection
    bpy.context.collection.objects.link(light_object)
    
    # Set the rotation (rotate 90 degrees around X axis)
    light_object.rotation_euler = (math.radians(90), 0, 0)
    
    print("Lighting setup complete")


def setup_render_settings(output_path, resolution_percentage=50, samples=4):
    """
    Configure render settings for fast rendering.
    
    Args:
        output_path: Path to output video file
        resolution_percentage: Resolution percentage (lower = faster)
        samples: Render samples (lower = faster)
    """
    # Fastest Render Settings


    # This disables shadow sampling in EEVEE (same as unchecking the 'Shadows' box in Sampling)
    bpy.context.scene.eevee.use_gtao = False  # Optional: also turn off ambient occlusion if desired
    bpy.context.scene.eevee.use_shadows = False

    bpy.context.scene.render.resolution_percentage = resolution_percentage
    bpy.context.scene.eevee.taa_render_samples = samples
    
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.render.simplify_subdivision = 2
    
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'LOWEST'
    
    print(f"Render settings configured, output path: {output_path}")


# =============================================
# MAIN FUNCTION
# =============================================

def process_mocap_animation(
    keypoints_file,
    video_path,
    output_path,
    armature_name="metarig",
    mesh_name="Cube",
    fps=30,
    frame_spacing=1,
    joint_radius=0.05,
    render_resolution=50,
    render_samples=4
):
    """
    Main function to process motion capture data and create animation.
    
    Args:
        keypoints_file: Path to keypoints file
        video_path: Path to background video
        output_path: Path to output video file
        armature_name: Name of the armature object
        mesh_name: Name of the mesh object
        fps: Frames per second for animation
        frame_spacing: Number of frames between keyframes
        joint_radius: Radius of the tracking spheres
        render_resolution: Resolution percentage for rendering
        render_samples: Number of samples for rendering
    """
    print(f"\n===== STARTING MOTION CAPTURE PROCESSING =====")
    print(f"Keypoints file: {keypoints_file}")
    print(f"Video path: {video_path}")
    print(f"Output path: {output_path}")
    
    # 1. Setup scene
    setup_scene(fps=fps)
    
    # 2. Load keypoints
    selected_indices = list(range(0, 13))
    p3ds = read_keypoints(keypoints_file, selected_indices)
    print(f"Loaded {len(p3ds)} frames of keypoints data")
    
    # 3. Create tracking objects
    joints = create_tracking_objects(p3ds, joint_radius=joint_radius)
    
    # 4. Animate joints
    total_frames = animate_joints(p3ds, joints, frame_spacing=frame_spacing)
    
    # 5. Setup armature and rigging
    scale_armature(p3ds, armature_name=armature_name)
    setup_armature_tracking(armature_name=armature_name, mesh_name=mesh_name)
    setup_spine_tracking(p3ds, armature_name=armature_name)
    setup_bone_constraints(armature_name=armature_name)
    
    # 6. Setup spine rotation
    setup_spine_rotation_handler()
    
    # 7. Setup camera and background
    camera = setup_camera()
    setup_background_video(camera, video_path)
    
    # 8. Setup compositing and lighting
    setup_compositing(video_path, total_frames)
    setup_lighting()
    
    # 9. Setup render settings
    setup_render_settings(output_path, resolution_percentage=render_resolution, samples=render_samples)
    
    # 10. Hide tracking objects
    setup_visibility(keep_visible=mesh_name)
    
    # 11. Bake spine rotation for rendering
    bake_spine_rotation()
    
    print(f"\n===== MOTION CAPTURE PROCESSING COMPLETE =====")
    print(f"Total frames: {total_frames}")
    print(f"Output will be saved to: {output_path}")
    
    return total_frames


# =============================================
# COMMAND LINE INTERFACE
# =============================================

import bpy
import argparse
import sys

def parse_arguments():
    """
    Parse command line arguments passed to the Blender script after '--'.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process motion capture data in Blender')

    parser.add_argument('--keypoints', help='Path to keypoints file')
    parser.add_argument('--video', help='Path to background video')
    parser.add_argument('--output', help='Path to output video file')
    parser.add_argument('--armature', default='metarig', help='Name of armature object')
    parser.add_argument('--mesh', default='Cube', help='Name of mesh object')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--frame-spacing', type=int, default=1, help='Number of frames between keyframes')
    parser.add_argument('--joint-radius', type=float, default=0.05, help='Radius of joint tracking objects')
    parser.add_argument('--resolution', type=int, default=50, help='Render resolution percentage')
    parser.add_argument('--samples', type=int, default=2, help='Render samples')
    parser.add_argument('--render', action='store_true', help='Render animation')

    # Only parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []  # No args provided

    return parser.parse_args(argv)


# =============================================
# MAIN ENTRY POINT
# =============================================


if __name__ == "__main__":
    if bpy.context.scene is not None:
        args = parse_arguments()

        # Resolve all file paths relative to the .blend file
        blend_dir = os.path.dirname(bpy.data.filepath)

        keypoints_path = os.path.join(blend_dir, args.keypoints)
        video_path = os.path.join(blend_dir, args.video)
        output_path = os.path.join(blend_dir, args.output)

        total_frames = process_mocap_animation(
            keypoints_file=keypoints_path,
            video_path=video_path,
            output_path=output_path,
            armature_name=args.armature,
            mesh_name=args.mesh,
            fps=args.fps,
            frame_spacing=args.frame_spacing,
            joint_radius=args.joint_radius,
            render_resolution=args.resolution,
            render_samples=args.samples
        )

        if args.render:
            print("\n===== STARTING RENDER =====")
            bpy.ops.render.render(animation=True)
            print("\n===== RENDER COMPLETE =====")
            print(f"Output saved to: {output_path}")
