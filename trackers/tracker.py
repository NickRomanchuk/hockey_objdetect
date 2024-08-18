from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import os
import sys
import cv2
import pandas as pd
from utils import batch_size

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_positions_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = get_center_of_bbox(bbox)
                    tracks[object][frame_num][track_id]['position'] = position


    def interpolate_puck_positions(self, puck_positions):
        puck_positions = [x.get(1, {}).get('bbox', []) for x in puck_positions]
        df_puck_positions = pd.DataFrame(puck_positions, columns=['x1','y1','x2','y2'])

        # interpolate missing values
        df_puck_positions = df_puck_positions.interpolate()
        df_puck_positions = df_puck_positions.bfill() # in case the first frame is missing

        puck_positions = [{1: {"bbox": x}} for x in df_puck_positions.to_numpy().tolist()]

        return puck_positions
    
    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # If file containing object detections exists, use that file as data
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Takes array of frames, and returns array of YOLO object detections
        detections = self.detect_frames(frames)

        tracks = {"players":[], "referees":[], "puck":[], "goalies":[]}
        for frame_num, detection in enumerate(detections):           
            cls_names_inv = {v:k for k,v in detection.names.items()}

            # Convert to supervision detection format and track objects
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["puck"].append({})
            tracks["goalies"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['goalie']:
                    tracks["goalies"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['puck']:
                    tracks["puck"][frame_num][1] = {"bbox":bbox}
        
        # If there isn't already a saved object detection file, we will save the object detections
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_box(self, frame, bbox, color):
        cv2.rectangle(frame,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      color,
                      2)

        return frame
    
    def draw_ellipse(self, frame, bbox, color):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=4,
            lineType=cv2.LINE_4
        )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, -1)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            puck_dict = tracks["puck"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalie_dict = tracks["goalies"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,0))
                frame = self.draw_ellipse(frame, player["bbox"], color)

            # Draw Goalie
            for track_id, goalie in goalie_dict.items():
                frame = self.draw_ellipse(frame, goalie["bbox"], (0,255,0))

            # Draw Puck
            #for _, puck in puck_dict.items():
            #    frame = self.draw_triangle(frame, puck["bbox"], (0,0,0))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            output_video_frames.append(frame)
        
        return output_video_frames
    
    def save_detections(self, video_frames, tracks):
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Draw Players
            for _, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0,0,0))
                frame = self.draw_box(frame, player["bbox"], color)

            # Draw Goalie
            for _, goalie in tracks["goalies"][frame_num].items():
                frame = self.draw_box(frame, goalie["bbox"], (0,255,0))

            # Draw Referee
            for _, puck in tracks["referees"][frame_num].items():
                frame = self.draw_box(frame, puck["bbox"], (0,0,0))

            # Draw Puck
            for _, referee in tracks["puck"][frame_num].items():
                frame = self.draw_box(frame, referee["bbox"], (0,255,255))

            cv2.imwrite(f'output_videos/bbox_frames/frame_{frame_num}.jpg',frame)

    def save_bboximages(self, video_frames, tracks):
        # save cropped image of players from first frame
        i = 0
        for _, player in tracks['players'][0].items():
            i += 1
            bbox = player['bbox']
            frame = video_frames[0]

            # crop bbox from frame
            cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            # save the cropped image
            cv2.imwrite(f'output_videos/trimmed_boxes/cropped_img_{i}.jpg', cropped_image)