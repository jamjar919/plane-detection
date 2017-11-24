import cv2
import os
import numpy as np
import random
import math
import csv
import functions;

# Constants
master_path_to_dataset = "TTBB-durham-02-10-17-sub10";
outdir = "output";
directory_to_cycle_left = "left-images";
directory_to_cycle_right = "right-images";
pause_playback = False;

# Custom constants
xoffset = 0;
yoffset = 0;
ransac_trials = 2000;
plane_fuzz = 0.01;
debug = True;
crop_disparity = True; # display full or cropped disparity image
objectResolution = 50;
obstacleRatio = 5;
obstacleRatioRecursive = 3;

# Always crop the side of disparity
xoffset = (135+xoffset)

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);
left_file_list = sorted(os.listdir(full_path_directory_left));

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    lastAbc, lastD = [], 0;
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # Read left and right images
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        #imgL = functions.correctLuminosity(imgL)

        # Convert to grayscale for processing
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        #####################
        ## PREPROCESS IMAGES
        #####################

        # Equalise histogram
        grayL = cv2.equalizeHist(grayL);
        grayR = cv2.equalizeHist(grayR);

        # Compute disparity image from undistorted and rectified stereo images
        disparity = stereoProcessor.compute(grayL,grayR);

        # Crop to reduce computation time
        croppedImage = imgL;
        if (crop_disparity):
            width = np.size(disparity, 1);
            height = np.size(disparity, 0);
            croppedImage = imgL[yoffset:390, xoffset:width]
            disparity = disparity[yoffset:390,xoffset:(width)];


        # Filter out noise and speckles (adjust parameters as needed)
        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # Correct the disparity
        # Fill in holes (value < 0) with the average of the row
        # This works well to preserve horizontal gradient
        for i in range(len(disparity)):
            row = disparity[i];
            a = np.average(row);
            for j in range(len(row)):
                if (disparity[i][j] < 1):
                    disparity[i][j] = a;

        # Thresh to remove too high values
        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);

        #####################
        ## RANSAC
        #####################
        sizeX, sizeY = (np.size(disparity, 0), np.size(disparity, 1));
        bestPoints = 0;
        abcBest, dBest = [0, 0, 0], 0;

        if (debug):
            print("turning points to 3D.... ")
        
        # Project points to 3D
        points = np.array(functions.project_disparity_to_3d(disparity, max_disparity));

        if (debug):
            print("done")
            print("doing ransac trials...")

        for i in range(0, ransac_trials):
            try:
                # Select three non colinear points
                cross_product_check = np.array([0,0,0]);
                while cross_product_check[0] == 0 and cross_product_check[1] == 0 and cross_product_check[2] == 0:
                    #[P1, P2, P3] = functions.getRandomPoints(disparity, max_disparity);
                    [P1,P2,P3] = points[random.sample(range(len(points)), 3)];
                    cross_product_check = np.cross(P1-P2, P2-P3);

                # Obtain plane coefficents
                coefficients_abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))
                coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0]+coefficients_abc[1]*coefficients_abc[1]+coefficients_abc[2]*coefficients_abc[2])
            
                # Check how many points are on plane
                distances = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d);
                
                # Convert to binary list and sum 
                # To quickly count the distances
                matchingPointsNum = (distances < plane_fuzz).sum()
                if matchingPointsNum > bestPoints:
                    bestPoints = matchingPointsNum;
                    abcBest = coefficients_abc;
                    dBest = coefficient_d;

            except ValueError as e: 
                # Sometimes the colinear thing screws up
                print("error selecting points");
                print(e);
            except np.linalg.linalg.LinAlgError as e:
                # Sometimes the linear algebra makes a singular matrix
                print("linalg error");
                print(e)

        # Try the last best plane match
        # This was a cool idea for an optimisation 
        # But it doesn't work very often
        if (len(lastAbc) and lastD):
            distances = abs((np.dot(points, lastAbc) - 1)/lastD);
            matchingPointsNum = (distances < plane_fuzz).sum()
            if matchingPointsNum > bestPoints:
                bestPoints = matchingPointsNum;
                abcBest = coefficients_abc;
                dBest = coefficient_d;
                if debug:
                    print("selecting previous plane")
        if debug:
            print("done...");

        # Normalise vector to indicate direction
        v = np.append(np.array(dBest), abcBest)
        factor = np.min(np.abs(v));
        direction = abcBest/factor;

        # Get the actual points on the plane
        matchingPoints = [];
        for point in points: 
            distance = abs(np.dot(point, abcBest) -1)/dBest;
            if (distance < plane_fuzz):
                matchingPoints.append(point);

        #######################
        # OBJECT DETECTION
        #######################

        if debug:
            print("doing object detection...");

        maxX, maxY, maxZ = np.max(points, 0);
        minX, minY, minZ = np.min(points, 0);
        cellSizeZ = (maxZ + abs(minZ))/objectResolution;
        cellSizeX = (maxX + abs(minX))/objectResolution;

        # Count number of points at each z index for average road density
        roadDensity = np.zeros(objectResolution, np.uint8)
        for i in range(1, len(matchingPoints)):
            ZNorm = matchingPoints[i][2]/(maxZ + abs(minZ));
            cellNum = math.floor(ZNorm*(objectResolution -1));
            roadDensity[cellNum] += 1;

        # Init variables
        elevationMap = np.zeros((objectResolution, objectResolution), np.uint8)
        pointMap = [];
        for y in range(0, objectResolution):
            pointMap.append([])
            for x in range(0, objectResolution):
                pointMap[y].append([]);

        # Loop and count number of points at each cell address
        for i in range(1, len(points)):
            ZNorm = points[i][2]/(maxZ + abs(minZ));
            XNorm = points[i][0]/(maxX + abs(minX));
            cellY = math.floor(ZNorm*(objectResolution -1));
            cellX = math.floor(XNorm*(objectResolution -1));
            elevationMap[cellY][cellX] += 1
            pointMap[cellY][cellX].append(points[i]);

        # Check actual density vs expected density of the road at that Z index
        elevationMapCopy = cv2.cvtColor(elevationMap,cv2.COLOR_GRAY2RGB);
        obstaclePoints = [];
        for y in range(0, len(elevationMap)):
            for x in range(0, len(elevationMap[y])):
                density = elevationMap[y][x];
                # Only add if we have pixel values in that z index
                if (roadDensity[y] != 0):
                    quotient = density/roadDensity[y]
                    # Compare the ratio of pixel values to the preset value (Paper says 6 is good)
                    if (quotient > obstacleRatio):
                        # Calculate the distance between the plane and the point
                        distance = abs(np.dot(np.average(pointMap[y][x], 0), abcBest) -1)/dBest;
                        if (distance > 0.07):
                            # Visualise the map
                            elevationMapCopy[y][x] = (255, 0, 0);
                            # Recursively add points to the map with a lower thresh
                            elevationMapCopy = functions.fillObstruction(elevationMapCopy, elevationMap, (x,y), roadDensity, obstacleRatioRecursive);
                        else:
                            elevationMapCopy[y][x] = (0, 0, 255);
        # Add relevant pixels to the render list
        for y in range(0, len(elevationMap)):
            for x in range(0, len(elevationMap[y])):
                if (elevationMapCopy[y][x][0] == 255):
                    obstaclePoints = obstaclePoints + pointMap[y][x];

        obstaclePoints2D = functions.project_3D_points_to_2D_image_points(obstaclePoints);
        cv2.imshow("elevation", elevationMapCopy)

        if debug:
            print("done");
            print("getting plane + projecting to 2D...")

        
        # Save params for plane test next time
        lastAbc = abcBest;
        lastD = dBest;
        
        # Project back to 2D
        matchingPoints2D = np.array(functions.project_3D_points_to_2D_image_points(matchingPoints));

        # Find biggest contour in our data
        # This is to remove noise from our raw data
        mask = np.zeros(disparity.shape, np.uint8);
        for i in range(0, len(matchingPoints2D)):
            point = matchingPoints2D[i];
            mask[point[1]][point[0]] = 255;

        # Apply dilation to close 1px horizontal lines in the mask
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("mask", mask);

        # Sort contours using python magic, and retrieve largest
        areas = [];
        _, contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        sortedContours = sorted(contours, key=cv2.contourArea, reverse=True);
        largest = sortedContours[0];

        mask = np.zeros(imgL.shape, np.uint8);
        disparity = (disparity / 16.).astype(np.uint8);
        disparityCopy = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB);

        cv2.drawContours(disparityCopy, [largest], 0, (0, 127, 0), 1)
        cv2.drawContours(mask, [largest], 0, (0, 127, 0), -1, 8, h, 0, (xoffset, yoffset))

        #http://users.utcluj.ro/~onigaf/files/pdfs/oniga_road_surface_ITSC2007.pdf


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
        if debug:
            print("done")
            print("normal", abcBest)

        print("normalised", direction)

        if debug:
            print("drawing...")

        imgL = cv2.add(imgL, mask);

        for point in obstaclePoints2D:
            point = (point[0] + xoffset, point[1] + yoffset);
            functions.markPoint(imgL, point, (0, 0, 255), 1)


        # hull = cv2.convexHull(matchingPoints2D);
        # if crop_disparity:
        #     for i in range(0, len(hull)):
        #         hull[i][0][0] += xoffset;
        # functions.poly(imgL, hull, (0, 0, 255), 1)

        # for i in range(0, len(matchingPoints2D)):
        #    markPoint(disparityCopy, (matchingPoints2D[i][0], matchingPoints2D[i][1]), (255, 0, 0))

        if debug:
            print("done")
        cv2.imshow('left image',imgL)
        cv2.imshow("disparity", disparityCopy);
        cv2.waitKey(10);
        cv2.imwrite(os.path.join(outdir, filename_left), imgL)

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
