from utils import readVideo, saveVideo
from trackers import Tracker

def main():
    # Read Video
    videoFrames = readVideo("inputVideos/15sec_input_720p.mp4")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.getObjectTracks(videoFrames)

    # Draw Output Annotations
    outputVideoFrames = tracker.drawAnnotations(videoFrames, tracks)

    # Save Video
    saveVideo(outputVideoFrames, "outputVideos/outputVideo.avi")

if __name__ == '__main__':
    main()