import cv2
import os
import numpy as np
import random;

#constants
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"; # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed
crop_disparity = True; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image
ransac_trials = 10;
plane_fuzz = 0.25;

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);
left_file_list = sorted(os.listdir(full_path_directory_left));

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

def markPoint(image, point, color):
    cv2.circle(image,point, 3, color, -1)

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
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        cv2.imshow('left image',imgL)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
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
        
        '''for i in range(len(disparity)):
            row = disparity[i];
            for j in range(len(row)):
                goodpixel = 255*16;
                pixel = disparity[i][j];
                if (pixel < 0):
                    disparity[i][j] = goodpixel;
                else:
                    goodpixel = pixel;'''

        # filter out noise and speckles (adjust parameters as needed)
        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        if (crop_disparity):
            width = np.size(disparity, 1);
            disparity = disparity[0:390,135:width];

        # RANSAC
        for i in range(0, ransac_trials):
            # Pick four points
            sizeX, sizeY = (np.size(disparity, 0), np.size(disparity, 1));
            sizeXFuzz = sizeX*plane_fuzz;
            sizeYFuzz = sizeY*plane_fuzz;
            # Pick a random point
            tl = (
                random.randint(0, sizeX),
                random.randint(0, sizeY)
            ); #topleft
            # Pick a random point to the right of tl
            tr = (
                random.randint(tl[0], sizeX),
                tl[1]
            ); #topright
            # Pick a random point below tl, with x coord in some region of tl's
            bl = (
                random.randint(0, tl[0]),
                random.randint(tl[1], sizeY)
            ) #bottomleft
            br = (
                random.randint(tr[0], sizeX),
                bl[1]
            ) #bottomright
            _, disparityCopy = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
            disparityCopy = (disparityCopy / 16.).astype(np.uint8);
            disparityCopy = cv2.cvtColor(disparityCopy, cv2.COLOR_GRAY2RGB);
            markPoint(disparityCopy, tl, 255*16)
            markPoint(disparityCopy, tr, 255*16)
            markPoint(disparityCopy, bl, 255*16)
            markPoint(disparityCopy, br, 255*16)
            
            cv2.imshow("disparity", disparityCopy);
            cv2.waitKey(300);

        # scale the disparity to 8-bit for viewing
        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);

        # crop disparity to chop out left part where there are with no disparity
        # as this area is not seen by both cameras and also
        # chop out the bottom area (where we see the front of car bonnet)    
    
        # cv2.imshow("disparity", disparity_scaled);

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
