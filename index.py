import cv2
import os
import numpy as np
import random
import math
import csv
import functions;
import time;



# Constants

# Paths
master_path_to_dataset = "TTBB-durham-02-10-17-sub10";
outdir = "output";
directory_to_cycle_left = "left-images";
directory_to_cycle_right = "right-images";

# General
debug = False; # Console logs steps of the program
writeImage = False; # Writes image to "outdir" directory
pause_playback = False;
show_images = True;

# Disparity
max_disparity = 128;
crop_disparity = True; # display full or cropped disparity image
xoffset = 0;
yoffset = 100;

# RANSAC
ransac_trials = 2000; # Number of trials to run for ransac
plane_fuzz = 0.01; # How far away from the plane should we allow points to be?

# Obstacle Detection
doObstacleDetection = True;
objectResolution = 450; # Number of cells in the top down view
roadDensityWindowSize = 10; # Size of the "rolling window" for the averaging
obstacleRatio = 3; # Ratio for detecting obstacles
obstacleRatioRecursive = 1; # Ratio for adding to obstacles

# Drawing
redLineMode = False; # Whether to just draw the simplified red polygon
                     # This also disables object detection


# Now begins the actual program...



# Turn off obstacle detection if red line mode is enabled
doObstacleDetection = doObstacleDetection and (not redLineMode);

# Always crop the side of disparity
xoffset = (135+xoffset)

# Actually sort out paths
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);
left_file_list = sorted(os.listdir(full_path_directory_left));

stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    print(full_path_filename_left);
    print(full_path_filename_right);

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

        # Gamma correct images
        grayL = functions.gamma(grayL, 2, 8);
        grayR = functions.gamma(grayR, 2, 8);

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
        dispNoiseFilter = 5;
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

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
            start = time.time()

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
            
                # Check normal is mainly in Y direction
                v = np.array(coefficients_abc)
                factor = np.max(np.abs(v));
                normal = coefficients_abc/factor;
                if normal[1] == 1:
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
            end = time.time()
            print("time:",end - start)
            print("done...");
            print("getting plane points...");

        # Normalise vector
        v = np.array(abcBest)
        factor = np.max(np.abs(v));
        normal = abcBest/factor;

        # Get the actual points on the plane
        matchingPoints = [];
        for point in points: 
            distance = abs(np.dot(point, abcBest) -1)/dBest;
            if (distance < plane_fuzz):
                matchingPoints.append(point);

        # Get average point and draw normal
        centrePoint = np.average(matchingPoints, 0);
        # Scale normal so it fits on screen
        normalScaled = normal/10;
        normalDir = [normalScaled[0] - centrePoint[0], normalScaled[1] - centrePoint[1], normalScaled[2] - centrePoint[2]];
        centrePoint2D = functions.projectPointTo2D(centrePoint);
        normalDir2D = functions.projectPointTo2D(normalDir);
        # Offset the points because of cropping
        centrePoint2D = (centrePoint2D[0] + xoffset, centrePoint2D[1] + yoffset);
        normalDir2D = (normalDir2D[0] + xoffset, normalDir2D[1] + yoffset);

        #######################
        # OBJECT DETECTION
        #######################

        if debug:
            print("done")
            print("doing object detection...");

        if doObstacleDetection:

            # We want to consider the full range of X coordinates
            # But only obstacles up to the end of our plane
            # So scale accordingly
            maxX, maxY, maxZ = np.max(points, 0);
            minX, minY, minZ = np.min(points, 0);
            maxXm, maxYm, maxZm = np.max(matchingPoints, 0);
            minXm, minYm, minZm = np.min(matchingPoints, 0);
            cellSizeZ = (maxZm + abs(minZm))/objectResolution;
            cellSizeX = (maxX + abs(minX))/objectResolution;

            # Count number of points at each z index for average road density
            roadDensity = np.zeros(objectResolution, np.uint16)
            for i in range(1, len(matchingPoints)):
                ZNorm = matchingPoints[i][2]/(maxZm + abs(minZm));
                cellNum = math.floor(ZNorm*(objectResolution -1));
                roadDensity[cellNum] += 1;

            # Eliminate leading 0's and replace with high value to simulate fade in
            roadDensity = np.trim_zeros(roadDensity, 'f')
            missing = np.dot(np.max(roadDensity),np.ones(objectResolution - len(roadDensity), np.uint16));
            roadDensity = np.concatenate([missing, roadDensity])

            # Equalise the road density using a rolling average
            # This is what I think they mean by Adaptive equalising in the paper
            N = roadDensityWindowSize;
            elements = [];
            for i in range(len(roadDensity)-1, -1, -1):
                elements.append(roadDensity[i]);
                roadDensity[i] = np.average(elements);
                if (len(elements) > N):
                    elements = elements[1:]

            # Init variables
            densityMap = np.zeros((objectResolution, objectResolution), np.uint16)
            pointMap = [];
            for y in range(0, objectResolution):
                pointMap.append([])
                for x in range(0, objectResolution):
                    pointMap[y].append([]);

            # Loop and count number of points at each cell address
            for i in range(1, len(points)):
                # Only count points in the same range as our plane
                if (points[i][2] <= maxZm):
                    ZNorm = points[i][2]/(maxZm + abs(minZm));
                    XNorm = points[i][0]/(maxX + abs(minX));
                    cellY = math.floor(ZNorm*(objectResolution -1));
                    cellX = math.floor(XNorm*(objectResolution -1));
                    densityMap[cellY][cellX] += 1
                    pointMap[cellY][cellX].append(points[i]);

            # Scale to view
            densityMapC = (densityMap / 16.).astype(np.uint8);
            densityMapC = cv2.equalizeHist(densityMapC);
            # Check actual density vs expected density of the road at that Z index
            densityMapC = cv2.cvtColor(densityMapC,cv2.COLOR_GRAY2RGB);
            obstaclePoints = [];
            for y in range(0, len(densityMap)):
                for x in range(0, len(densityMap[y])):
                    density = densityMap[y][x];
                    # Only add if we have pixel values in that z index
                    if (roadDensity[y] != 0):
                        quotient = density/roadDensity[y]
                        # Compare the ratio of pixel values to the preset value (Paper says 6 is good)
                        if (quotient > obstacleRatio):
                            # Calculate the distance between the plane and the point
                            distance = abs(np.dot(pointMap[y][x], abcBest) -1)/dBest;
                            if np.all(distance > 0.05):
                                # Visualise the map
                                densityMapC[y][x] = (255, 0, 0);
                                # Recursively add points to the map with a lower thresh
                                densityMapC = functions.fillObstruction(densityMapC, densityMap, (x,y), roadDensity, obstacleRatioRecursive);
                            else:
                                densityMapC[y][x] = (0, 0, 255);
            # Add relevant pixels to the render list
            for y in range(0, len(densityMap)):
                for x in range(0, len(densityMap[y])):
                    if (densityMapC[y][x][0] == 255):
                        obstaclePoints = obstaclePoints + pointMap[y][x];

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

        # If red line mode is enabled smooth the edges to make a dumb polygon
        if redLineMode:
            kernel = np.ones((3,3),np.uint8)
            for i in range(0, 10):
                mask = cv2.erode(mask, kernel)
                mask = cv2.GaussianBlur(mask, (15, 15), 0);
                cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY, mask);

        # Sort contours using python magic, and retrieve largest
        areas = [];
        _, contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        sortedContours = sorted(contours, key=cv2.contourArea, reverse=True);
        largest = sortedContours[0];

        mask = np.zeros(imgL.shape, np.uint8);
        disparity = (disparity / 16.).astype(np.uint8);
        disparityCopy = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB);

        cv2.drawContours(disparityCopy, [largest], 0, (0, 127, 0), 1)
        if not redLineMode:
            cv2.drawContours(mask, [largest], 0, (127, 127, 127), -1, 8, h, 0, (xoffset, yoffset))

        #http://users.utcluj.ro/~onigaf/files/pdfs/oniga_road_surface_ITSC2007.pdf

        if debug:
            print("done")

        print("road surface normal: (", normal[0][0], ",",normal[1][0],",",normal[2][0],")")
        print();

        if debug:
            print("drawing...")

        if doObstacleDetection:
            obstacleMask = np.zeros(imgL.shape, np.uint8);
            for point in obstaclePoints:
                # Consider the infinite line defined by the 3D point and the 3D point plus the normal
                # We want to get the intersection of the line with the plane so we can draw the obstacles at real height
                PX, PY, PZ = point;
                v = normal/0.001;
                NX, NY, NZ = (PX + v[0],PY + v[1], PZ + v[2]);
                a, b, c = abcBest;
                d = dBest;
                # Plane is defined by ax + by + cz = d
                # Line defined by parametric eq's 
                # x = PX + tNX, y = PY + tNY, z = PZ + tNZ
                # Solve for t (I worked out this equation)
                t = (d-(a*PX)-(b*PY)-(c*PZ))/((a*NX) + (b*NY) + (c*NZ));
                # Plug into parametric equation
                intersection = (PX + NX*t, PY + NY*t, PZ + NZ*t);
                # Translate to 2D points
                point2D = functions.projectPointTo2D(point);
                intersection2D = functions.projectPointTo2D(intersection);
                # Compensate for offset
                point2D = (point2D[0] + xoffset, point2D[1] + yoffset);
                intersection2D = (intersection2D[0] + xoffset, intersection2D[1] + yoffset);
                # Draw polyline
                functions.poly(obstacleMask, np.array([point2D, intersection2D]), (127, 127, 127), 1)

        # Compose image
        if doObstacleDetection:
            # Mask the obstacles with the plane image
            obstacleMask = cv2.subtract(obstacleMask, mask);
            obstacleMask[:,:,0] = 0; # blue channel
            obstacleMask[:,:,1] = 0; # green channel
            # Zero all pixels after the crop
            for y in range(390, len(obstacleMask)):
                for x in range(0, len(obstacleMask[y])):
                    obstacleMask[y][x] = (0, 0, 0);
            imgL = cv2.add(imgL, obstacleMask);
            # Draw points
            maxX, maxY, maxZ = np.max(obstaclePoints, 0);
            for point in obstaclePoints:
                # Scale color according to distance from camera
                color = math.floor(255-((point[2])/maxZ)*255);
                point2D = functions.projectPointTo2D(point);
                point2D = (point2D[0] + xoffset, point2D[1] + yoffset)
                functions.markPoint(imgL, point2D, (0,0,color),1)

        # draw plane outline
        if redLineMode:
            hull = cv2.convexHull(largest);
            for point in hull:
                point[0] = (point[0][0] + xoffset, point[0][1] + yoffset)
            functions.poly(imgL, hull, (0, 0, 255), 3);
        else:
            # draw plane
            mask[:,:,0] = 0; # blue channel
            mask[:,:,2] = 0; # red channel
            imgL = cv2.add(imgL, mask);
            cv2.drawContours(imgL, [largest], 0, (0, 127, 0), 3, 8, h, 0, (xoffset, yoffset))

        # Draw the normal
        functions.markPoint(imgL, centrePoint2D, (255, 0 , 0), 4)
        functions.poly(imgL, np.array([centrePoint2D, normalDir2D]), (255, 0, 0), 4)
        arrowLeft = (normalDir2D[0] + 10, normalDir2D[1] + 10);
        arrowRight = (normalDir2D[0] - 10, normalDir2D[1] + 10);
        functions.poly(imgL, np.array([arrowLeft, normalDir2D]), (255, 0, 0), 4)
        functions.poly(imgL, np.array([arrowRight, normalDir2D]), (255, 0, 0), 4)

        # Combine into one image
        if doObstacleDetection:
            doubleDense = functions.combineImagesH(densityMapC, densityMapC);
            stack = functions.combineImagesV(disparityCopy, doubleDense)
            output = functions.combineImagesH(imgL, stack);
        else:
            output = functions.combineImagesH(imgL, disparityCopy);

        if debug:
            print("done")
        if show_images:
            cv2.imshow("output",output)
            cv2.waitKey(10);
        if writeImage:
            cv2.imwrite(os.path.join(outdir, filename_left), output)

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
