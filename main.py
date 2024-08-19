from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedDistanceEstimator
import cv2

def main():
    # Initialize Tracker, uses best.pt model
    tracker = Tracker('models/best.pt')

    # Read video, returns array of frames (array of pixels)
    video_frames = read_video('input_videos/input_video.mp4')

    # Save each of the video frames
    for index, frame in enumerate(video_frames):
        cv2.imwrite(f'output_videos/raw_frames/frame_{index}.jpg',frame)

    # Get the grouped detections
    detections = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pk1')
    
    # Get object positions
    tracker.add_positions_to_tracks(detections)

    # camera movement estimator
    camera_movement_esitmator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_esitmator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pk1')
    camera_movement_esitmator.add_adjust_postions_to_tracks(detections, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_tranfsormed_position_to_tracks(detections)

    # interpolate puck positions
    # detections["puck"] = tracker.interpolate_puck_positions(detections["puck"])
    
    # Speed and distance estimator
    #speed_and_distance_estimator = SpeedDistanceEstimator()
    #speed_and_distance_estimator.add_speed_and_distance_to_tracks(detections)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, detections)

    # Draw Camera Movement
    #output_video_frames = camera_movement_esitmator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw Speed and Distance
    #speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, detections)

    # Save each frame with bounding boxes
    #tracker.save_detections(video_frames, detections)

    # Save each bounding box as a seperate image
    #tracker.save_bboximages(video_frames, detections)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
    print('\n\nDone!\n\n')