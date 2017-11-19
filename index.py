import cv2
import os
import numpy as np
import random
import math
import csv
import functions;

#constants
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"; # ** need to edit this **
outdir = "output";
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed
pause_playback = False; # pause until key press after each image
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
image_centre_h = 262.0;
image_centre_w = 474.5;

xoffset = 0;
yoffset = 100;
ransac_trials = 2000;
plane_fuzz = 0.01;
debug = True;
crop_disparity = True; # display full or cropped disparity image

# Always crop the side of disparity
xoffset = (135+xoffset)

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);
left_file_list = sorted(os.listdir(full_path_directory_left));

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    Zmax = ((f * B) / 2);
    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index
            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = ((f * B) / disparity[y,x]);
                # Zmax = Z
                X = ((x - image_centre_w) * Zmax) / f;
                Y = ((y - image_centre_h) * Zmax) / f;

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

    # calc. Zmax as per above

    Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / Zmax) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Zmax) + image_centre_h;
        points2.append((math.floor(x),math.floor(y)));

    return points2;

#####################################################################

for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists
    lastAbc, lastD = [], 0;
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        #imgL = functions.correctLuminosity(imgL)
        #imgR = functions.correctLuminosity(imgR)

        # cv2.imshow('right image',imgR)
        # convert to grayscale for processing
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        


        ###
        ## PREPROCESS IMAGES
        ###

        grayL = cv2.equalizeHist(grayL);
        grayR = cv2.equalizeHist(grayR);
        kernel = (5,5);
        grayL = cv2.GaussianBlur(grayL, kernel, 0);
        grayR = cv2.GaussianBlur(grayR, kernel, 0);

        # compute disparity image from undistorted and rectified stereo images
        disparity = stereoProcessor.compute(grayL,grayR);
        croppedImage = imgL;
        if (crop_disparity):
            width = np.size(disparity, 1);
            height = np.size(disparity, 0);
            croppedImage = imgL[yoffset:390, xoffset:width]
            disparity = disparity[yoffset:390,xoffset:(width)];


        # filter out noise and speckles (adjust parameters as needed)
        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        #disparity = cv2.GaussianBlur(disparity, kernel, 0);
        for i in range(len(disparity)):
            row = disparity[i];
            s = 0;
            for j in range(len(row)):
                s += disparity[i][j];
            a = s/len(row);
            for j in range(len(row)):
                if (disparity[i][j] < 1):
                    disparity[i][j] = a #isparity[i][j-1];

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        #disparity = (disparity / 16.).astype(np.uint8);

        # RANSAC
        sizeX, sizeY = (np.size(disparity, 0), np.size(disparity, 1));
        if (debug):
            print("turning points to 3D.... ")
        points = np.array(project_disparity_to_3d(disparity, max_disparity));
        if (debug):
            print("done")
        bestPoints = 0;
        abcBest, dBest = [0, 0, 0], 0;
        print("doing ransac trials...")
        for i in range(0, ransac_trials):
            try:
                cross_product_check = np.array([0,0,0]);
                while cross_product_check[0] == 0 and cross_product_check[1] == 0 and cross_product_check[2] == 0:
                    [P1, P2, P3] = functions.getRandomPoints(disparity, max_disparity);
                    #[P1,P2,P3] = points[random.sample(range(len(points)), 3)];
                    # make sure they are non-collinear
                    cross_product_check = np.cross(P1-P2, P2-P3);

                # Obtain plane coefficents
                coefficients_abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))
                coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0]+coefficients_abc[1]*coefficients_abc[1]+coefficients_abc[2]*coefficients_abc[2])
            
                # Check how many points are on plane
                distances = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d);
                
                matchingPointsNum = (distances < plane_fuzz).sum()
                if matchingPointsNum > bestPoints:
                    bestPoints = matchingPointsNum;
                    abcBest = coefficients_abc;
                    dBest = coefficient_d;
            except ValueError as e: 
                print("error selecting points");
                print(e);
            except np.linalg.linalg.LinAlgError as e:
                print("linalg error");
                print(e)
        # try the last best plane match
        if (len(lastAbc) and lastD):
            distances = abs((np.dot(points, lastAbc) - 1)/lastD);
            matchingPointsNum = (distances < plane_fuzz).sum()
            if matchingPointsNum > bestPoints:
                bestPoints = matchingPointsNum;
                abcBest = coefficients_abc;
                dBest = coefficient_d;
                print("selecting previous plane")

        print("done...");

        # normalise vector to indicate direction
        v = np.append(np.array(dBest), abcBest)
        factor = np.min(np.abs(v));
        direction = abcBest/factor;
        print(np.max(points, 2))
        print("getting plane + projecting to 2D...")
        # Get the actual points
        matchingPoints = [];
        for point in points: 
            distance = abs(np.dot(point, abcBest) -1)/dBest;
            
            if (distance < plane_fuzz):
                matchingPoints.append(point);
        lastAbc = abcBest;
        lastD = dBest;
        
        #Project back to 2d
        matchingPoints2D = np.array(project_3D_points_to_2D_image_points(matchingPoints));

        #Find biggest contour
        mask = np.zeros(disparity.shape, np.uint8);
        for i in range(0, len(matchingPoints2D)):
            point = matchingPoints2D[i];
            mask[point[1]][point[0]] = 255;

        # Apply dilation to close horizontal lines
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("mask", mask);

        areas = [];
        _, contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        sortedContours = sorted(contours, key=cv2.contourArea, reverse=True);

        largest = sortedContours[0];

        mask = np.zeros(imgL.shape, np.uint8);
        disparity = (disparity / 16.).astype(np.uint8);
        disparityCopy = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB);

        cv2.drawContours(disparityCopy, [largest], 0, (0, 127, 0), 1)
        cv2.drawContours(mask, [largest], 0, (0, 127, 0), -1, 8, h, 0, (xoffset, yoffset))


        # Apply a vague floodfill to the contour based on average hue
        # print("applying floodfill...")
        # hsvImg = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV);
        # h, s, v = cv2.split(hsvImg);
        # colorSum = 0;
        # colorNum = 0;
        # for y in range(0, len(h)):
        #         for x in range(0, len(h[y])):
        #             if (mask[y][x][1] == 127):
        #                 colorSum += s[y][x];
        #                 colorNum += 1;
        # averageSat = colorSum/colorNum;
        # print(averageSat);
        print("done")
        print("normal", abcBest)
        print("normalised", direction)

        print("drawing...")

        imgL = cv2.add(imgL, mask);

        '''hull = cv2.convexHull(matchingPoints2D);
        if crop_disparity:
            for i in range(0, len(hull)):
                hull[i][0][0] += yoffset;
                hull[i][0][1] += xoffset;
        poly(imgL, hull, (0, 0, 255), 1)
'''
        # for i in range(0, len(matchingPoints2D)):
        #    markPoint(disparityCopy, (matchingPoints2D[i][0], matchingPoints2D[i][1]), (255, 0, 0))

        print("done")
        cv2.imshow('left image',imgL)
        cv2.imshow("disparity", disparityCopy);
        cv2.waitKey(10);
        cv2.imwrite(os.path.join(outdir, filename_left), imgL)
        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

cv2.destroyAllWindows()
