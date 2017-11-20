import numpy as np
import cv2
import math
import random

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
image_centre_h = 262.0;
image_centre_w = 474.5;

def correctLuminosity(img, mode=cv2.COLOR_BGR2Lab, moderev=cv2.COLOR_Lab2BGR):
    new = cv2.cvtColor(img, mode)
    # Get the L component
    l, a, b = cv2.split(new)
    # Apply clahe
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    cv2.merge((l, a, b), new)
    new = cv2.cvtColor(new, moderev)
    t = np.hstack((img, new));
    cv2.imshow("after", t)
    return new;

def projectPointTo3D(point, source, max_disparity, image_centre_h = 262.0, image_centre_w = 474.5):
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    x, y = point;
    Z = (f * B) / source[y][x];
    X = ((x - image_centre_w) * Z) / f;
    Y = ((y - image_centre_h) * Z) / f;
    return np.array([X,Y,Z]);

def markPoint(image, point, color):
    cv2.circle(image,point, 1, color, -1)

def poly(image, points, color, thickness=3):
    cv2.polylines(image, [points],True,color, thickness);

def getRandomPoints(disparity, max_disparity=128):
    fuzz = 0.25;
    height, width = disparity.shape[:2];
    hfuzz = np.floor(height*fuzz);
    wfuzz = np.floor(width*fuzz)
    P1 = (
        random.randint(0+wfuzz, width-wfuzz),
        random.randint(0+hfuzz, height-hfuzz)
    );
    P2 = (
        random.randint(0, P1[0]),
        random.randint(P1[1], height-1)
    );
    P3 = (
        random.randint(P1[0], width-1),
        P2[1]
    );
    # Project to 3D
    return [
        projectPointTo3D(P1, disparity, max_disparity),#, midh, midw),
        projectPointTo3D(P2, disparity, max_disparity),#, midh, midw),
        projectPointTo3D(P3, disparity, max_disparity),#, midh, midw)
    ];

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):
    points = [];
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    height, width = disparity.shape[:2];
    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index
            # if we have a valid non-zero disparity
            if (disparity[y,x] > 0):
                # calculate corresponding 3D point [X, Y, Z]
                Z = ((f * B) / disparity[y,x]);
                X = ((x - image_centre_w) * Z) / f;
                Y = ((y - image_centre_h) * Z) / f;
                # add to points

                if(len(rgb) > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);

    return np.array(points);

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = [];
    for i1 in range(len(points)):
        # reverse earlier projection for X and Y to get x and y again
        Z = points[i1][2];
        x = ((points[i1][0] * camera_focal_length_px) / Z) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Z) + image_centre_h;
        points2.append((math.floor(x),math.floor(y)));

    return points2;

#####################################################################