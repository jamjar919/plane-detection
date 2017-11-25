import cv2
import os
import math
from skvideo import VideoWriter

path = 'output'
outname = 'video.mp4'
path = os.path.join(path)

imageFiles = sorted(os.listdir(path));

print("reading shape...");
# Read first file to get shape data
first = os.path.join(path, imageFiles[0])
frame = cv2.imread(first)
height, width, channels = frame.shape
#fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
#out = cv2.VideoWriter(outname, fourcc, 25.0, (width, height))
out = VideoWriter(filename, frameSize=(width, height))
out.open()

print("writing file...");
for i in range(0, len(imageFiles)):
    if math.floor((i/len(imageFiles))*100) % 10 == 0:
        print("#", end="");

    image = os.path.join(path, imageFiles[i]);
    frame = cv2.imread(image);
    out.write(frame);

    # cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        print("quit")
        break
print(" done");
print("wrote video to", outname);
out.release()
cv2.destroyAllWindows()
