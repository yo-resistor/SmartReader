# Import the necessary packages
from imutils import contours
import imutils
import os
import cv2
import numpy as np
import csv

# Define the dictionary of digit segments so we can identify each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# Search image files in test folder
folder_path = "test"
filename_list = os.listdir(folder_path)

# Write a new csv file to save data
header = ['index', 'filename', 'value']
with open('data/'+folder_path+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

for idx_filename in range(len(filename_list)):
    img_filename = filename_list[idx_filename]
    img_original= cv2.imread(folder_path+'/'+img_filename, cv2.IMREAD_COLOR)
    
    # Preprocess the image by resizing it, converting it to graycale, blurring it, and computing an edge map
    img_resized = imutils.resize(img_original, height=700)
    img_cropped = img_resized[240:333, 330:620]
    img_grayscale = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gaussianblur = cv2.GaussianBlur(img_grayscale, (3, 3), 0)
    img_cannyedge = cv2.Canny(img_gaussianblur, 50, 200, 255)
    
    # Threshold the warped image, then apply a series of morphological operations to cleanup the thresholded image
    thresh = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = 255-cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the thresholded image, then initialize the digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    
    # Loop over the digit area candidates
    for c in cnts:
        # Compute the bounding box of the contours
        (x, y, w, h) = cv2.boundingRect(c)
        print('top')
        print(x, y, w, h)
        # If the contour is sufficiently large, it must be a digit
        if h > 55:
            if w < 27:
                min_pixel_width = 32
                x = x + w - min_pixel_width
                w = min_pixel_width
            digitCnts.append(c)
        print('top after')
        print(x, y, w, h)
            
    # Sort the contours from left-to-right, then initialize the actual digits themselves
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []
    
    img_bounding = img_cropped.copy()
    
    if len(digitCnts) == 6:
        # Loop over each of the digits
        for c in digitCnts:
            # Extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            print('bottom')
            print(x, y, w, h)
            if w < 27:
                min_pixel_width = 32
                x = x + w - min_pixel_width
                w = min_pixel_width
            print('bottom after')
            print(x, y, w, h)
            roi = thresh[y:y + h, x:x + w]
            # Compute the width and height of each of the 7 segments we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.3), int(roiH * 0.15))
            dHC = int(roiH * 0.1)
            # Define the set of 7 segments
            segments = [
        		((0, 0), (w, dH)),	# top
        		((0, 0), (dW, h // 2)),	# top-left
        		((w - dW, 0), (w, h // 2)),	# top-right
        		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
        		((0, h // 2), (dW, h)),	# bottom-left
        		((w - dW, h // 2), (w, h)),	# bottom-right
        		((0, h - dH), (w, h))	# bottom
        	]
            on = [0] * len(segments)
            image = cv2.rectangle(img_bounding, (x,y), (x+w,y+h), (255,0,0), 5)
            
            # Loop over the segments   
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # Extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                
                # If the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) > 0.5:
                    on[i]= 1
            
            # print(on)
            if on == [1, 1, 1, 0, 1, 1, 1] or \
               on == [0, 0, 1, 0, 0, 1, 0] or \
               on == [1, 0, 1, 1, 1, 0, 1] or \
               on == [1, 0, 1, 1, 0, 1, 1] or \
               on == [0, 1, 1, 1, 0, 1, 0] or \
               on == [1, 1, 0, 1, 0, 1, 1] or \
               on == [1, 1, 0, 1, 1, 1, 1] or \
               on == [1, 0, 1, 0, 0, 1, 0] or \
               on == [1, 1, 1, 1, 1, 1, 1] or \
               on == [1, 1, 1, 1, 0, 1, 1]:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                # print(on)
                # print('good')
            else:
                print(img_filename)
                # print(len(digits))
                break
            
        if len(digits) == 6:
            value = float(0)
            for idx in range(len(digits)):
                value += float(digits[idx])*(10**(1-idx))
                value = np.round(value, decimals = 4)
        else:
            value = float(-1)
            
    else:
        value = float(-1)
    
    print(value)
    
    # Write a cvs file
    with open('data/'+folder_path+'.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([idx_filename, img_filename, value])
        
    
    # Read the images
    cv2.imshow("Resized", img_resized)
    # cv2.waitKey()
    cv2.imshow("Cropped", img_cropped)
    # cv2.waitKey()
    # cv2.imshow("Grayscale", img_grayscale)
    # cv2.waitKey()
    # cv2.imshow("Gaussian Blur", img_gaussianblur)
    # cv2.waitKey()
    # cv2.imshow("Canny Edge", img_cannyedge)
    # cv2.waitKey()
    # cv2.imshow("Thresh", thresh)
    # cv2.waitKey()     
    cv2.imshow("Bounding box", image)
    cv2.waitKey()     
    
# close all open windows
cv2.destroyAllWindows()