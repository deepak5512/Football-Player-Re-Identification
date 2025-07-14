from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from utils import getCenterOfBbox, getBboxWidth

class Tracker:
    def __init__(self, modelPath):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()

    def detectFrames(self, frames):
        batchSize = 16
        detections = []

        # Get the prediction of frames in batches (Batch Size = 16)
        for i in range(0, len(frames), batchSize):
            batchDetection = self.model.predict(frames[i: i+batchSize], conf = 0.1)
            detections += batchDetection

        return detections

    def getObjectTracks(self, frames):

        detections = self.detectFrames(frames)

        # Tracks of Every Frame
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frameNum, detection in enumerate(detections):
            classNames = detection.names                              # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            classNamesInverse = {v: k for k, v in classNames.items()} # {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}

            # Convert to Supervision Detection Format
            detectionSupervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to Player Object
            for objectIndex, classID in enumerate(detectionSupervision.class_id):
                if classNames[classID] == "goalkeeper":
                    detectionSupervision.class_id[objectIndex] = classNamesInverse["player"]

            # Track Objects
            detectionWithTracks = self.tracker.update_with_detections(detectionSupervision) # 0 -> Bounding Box[[]], 1 -> Mask, 2 -> Confidence[], 3 -> Class ID[], 4 -> Track ID[]

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Add Every Frame in their corresponding track
            for frameDetection in detectionWithTracks:
                bbox = frameDetection[0].tolist()
                classID = frameDetection[3]
                trackID = frameDetection[4]

                # Adding Player Tracks
                if classID == classNamesInverse["player"]:
                    tracks["players"][frameNum][trackID] = {"bbox": bbox}
                
                # Adding Referee Tracks
                if classID == classNamesInverse["referee"]:
                    tracks["referees"][frameNum][trackID] = {"bbox": bbox}

            # Adding Ball Tracks
            for frameDetection in detectionWithTracks:
                bbox = frameDetection[0].tolist()
                classID = frameDetection[3]

                if classID == classNamesInverse["ball"]:
                    tracks["ball"][frameNum][1] = {"bbox": bbox}

        return tracks

    def drawEllipse(self, frame, bbox, color, trackID = None):
        y2 = int(bbox[3])

        x_center, _ = getCenterOfBbox(bbox)
        width = getBboxWidth(bbox)

        # Draw the ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rectangle = x_center - rectangle_width//2
        x2_rectangle = x_center + rectangle_width//2
        y1_rectangle = (y2 - rectangle_height//2) + 15
        y2_rectangle = (y2 + rectangle_height//2) + 15

        # Write the Track ID for the player
        if trackID is not None:
            cv2.rectangle(
                frame,
                (int(x1_rectangle), int(y1_rectangle)),
                (int(x2_rectangle), int(y2_rectangle)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rectangle + 12
            if trackID > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{trackID}",
                (int(x1_text), int(y1_rectangle + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame
    
    def drawTriangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = getCenterOfBbox(bbox)

        # Coordinates of all the three points of Triangle
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        # Draw the triangle with blue color filled
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # Draw a black boundary around the triangle
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def drawAnnotations(self, videoFrames, tracks):
        outputVideoFrames = []

        # Loop through all the video frames
        for frameNum, frame in enumerate(videoFrames):
            frame = frame.copy()

            playerDict = tracks["players"][frameNum]
            ballDict = tracks["ball"][frameNum]
            refereeDict = tracks["referees"][frameNum]

            # Draw Players Annotations (Ellipse & Track ID)
            for trackID, player in playerDict.items():
                frame = self.drawEllipse(frame, player["bbox"], (0, 255, 0), trackID)

            # Draw Referees Annotations (Ellipse)
            for _, referee in refereeDict.items():
                frame = self.drawEllipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw Ball Annotations (Triangle)
            for trackID, ball in ballDict.items():
                frame = self.drawTriangle(frame, ball["bbox"], (255, 0, 0))

            outputVideoFrames.append(frame)

        return outputVideoFrames