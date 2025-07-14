## Football Player Re-Identification in a Single Feed

### Overview

This project implements a solution for player re-identification, ensuring that the same player retains the same ID, even after going out of view.

The task is that given a 15-second video, identify each player and ensure that players who go out of frame and reappear are assigned the same identity as before.

---

### Approach & Methodology

1. **Object Detection**
    - Used the **YOLOv8** model (via the `ultralytics` library) to detect:
        - `player`, `goalkeeper`, `referee`, and `ball`.
    - The model is loaded from a custom-trained weight file: `models/best.pt`.
2. **Object Tracking**
    - Used **ByteTrack** from the `supervision` library for multi-object tracking.
    - All detected `goalkeepers` are reclassified as `players` to unify tracking logic.
3. **Annotation Drawing**
    - **Ellipses** are drawn under players and referees.
    - **Triangles** are used to annotate the ball.
    - **Track IDs** are shown for players to allow re-identification over time.
4. **Video Processing**
    - `readVideo`: Extracts frames from input.
    - `saveVideo`: Combines annotated frames into output video.

---

### Techniques Tried & Outcomes

| Technique | Purpose | Outcome |
| --- | --- | --- |
| YOLOv8 | Fast object detection | Successfully detects all desired entities |
| ByteTrack | Robust ID assignment across frames | Maintains consistent track IDs for players |
| Custom Ellipse/Triangle Drawing | Visualization | Improves interpretability of movement and roles |

---

### Challenges Encountered

1. **Class Ambiguity**
    - `goalkeeper` and `player` are treated as distinct by YOLO but were unified for tracking. This required class reassignment logic.
2. **Batch Processing Limits**
    - YOLO inference is done in batches of 16. Larger videos may lead to memory issues or longer processing times.
3. **Assumption on Ball Tracking**
    - Only a single ball is tracked per frame (always assumed ID = 1). Multiple detections might cause inaccuracies.
4. **Track ID Reassignment**
    - In complex scenes with dense players, ByteTrack may reassign IDs. Visual continuity could be affected.

---

### Remaining Work & Future Scope

| Area | What's Missing | Next Steps |
| --- | --- | --- |
| ID Consistency | Re-ID in cases of dense players situation is not handled | Add appearance embeddings or jersey number OCR |
| Model Accuracy | Ball detection can fail at high speed due to model limitation | Ball Interpolation: In the frames where the ball is not detected, we can interpolate and assume that the ball moves in straight line |
| Scalability | Processing large videos can be time-consuming | Parallelize inference |

---

### Project Structure

```
Football-Player-Re-Identification/
├── main.py
├── trackers/
│   ├── __init__.py
│   └── tracker.py
├── utils/
│   ├── videoUtils.py
│   └── bbox.py
├── models/
│   └── best.pt
├── inputVideos/
│   └── 15sec_input_720p.mp4
└── outputVideos/
    └── outputVideo.avi
```
