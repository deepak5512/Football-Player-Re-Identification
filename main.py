from utils import readVideo, saveVideo
from trackers import Tracker

def main():
    modelPath = "models/best.pt"
    videoPath = "inputVideos/15sec_input_720p.mp4"

    # Read Video
    videoFrames = readVideo(videoPath)

    # Initialize Tracker
    tracker = Tracker(modelPath)

    tracks = tracker.getObjectTracks(videoFrames)

    # Draw Output Annotations
    outputVideoFrames = tracker.drawAnnotations(videoFrames, tracks)

    # Save Video
    saveVideo(outputVideoFrames, "outputVideos/outputVideo.avi")

if __name__ == '__main__':
    main()