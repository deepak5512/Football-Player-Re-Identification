import cv2

# Read the video and returns all the frames of that video
def readVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frames = []
    while True:
        flag, frame = cap.read()
        # If all the frames are completed flag will become false
        if not flag:
            break
        frames.append(frame)

    return frames


# Take the frames as input and merge them to create a video
def saveVideo(outputVideoFrames, outputVideoPath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputVideoPath, fourcc, 24, (outputVideoFrames[0].shape[1], outputVideoFrames[0].shape[0]))
    for frame in outputVideoFrames:
        out.write(frame)

    out.release()