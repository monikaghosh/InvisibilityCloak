import cv2
import time
import numpy as np

#To save the output in a file output.avi
#FourCC is a 4-byte code used to specify the video codec. The list of
# available codes can be found in fourcc.org. 
#codec is a device or program that compresses data to enable faster transmission 
#and decompresses received data.
fourcc = cv2.VideoWriter_fourcc(*'XVID') #to specify what should be our extension of the video file

#<VideoWriter object>	=	cv.VideoWriter(	filename, fourcc object, fps, frameSize)
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#Starting the webcam
cap = cv2.VideoCapture(0) # 0 for the system's camera, 1 for external

#Allowing the webcam to start by making the code sleep for 3 seconds 
time.sleep(3)
bg = 0

#We need to have a video that has some seconds dedicated to the background frame so that it could
#easily save the background image.
for i in range(60):
    #store the values in 2 vars --return will store true or false, bg will store the frame we are capturing
    ret, bg = cap.read() 
#Flipping the background -camera captures the image inverted
bg = np.flip(bg, axis=1)

#Reading the captured frame till the camera is open ---to read every frame from from the camera till the web cam is open
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #Flipping the image for consistency
    img = np.flip(img, axis=1)

    #Converting the color from BGR to HSV to detect the red color more effeciently
    #when we have to pay more attention to colors we use HSV mode
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Generating mask to detect red color
    #These values can also be changed as per the color
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255,255])
    mask_1 = cv2.inRange(hsv, lower_red, upper_red) #to detect the color between the given range

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_1 = mask_1 + mask_2 #the red mask

    #Open and expand the image where there is mask 1 (color)
    #https://www.tutorialspoint.com/opencv/opencv_morphological_operations.htm
    #morphologyEx(src, op, kernel)
    #mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

# Now we need to create a mask to
# segment out the red color from the
# frame.
# To do so we'll be using the
# bitwise_not() method.

    #Selecting only the part that does not have mask one and saving in mask 2 
    #to segment out the red 
    mask_2 = cv2.bitwise_not(mask_1)#mask2 is all color except red

# Now, we need to create 2 resolutions.
# First one would be an image without
# color red (or any other color that you
# choose) and the second one would be
# the background from the background
# image we captured earlier just for the
# parts where red color was (mask 1).

    #Keeping only the part of the images without the red color 
    #(or any other color you may choose)
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    #Keeping only the part of the images with the red color
    #(or any other color you may choose)
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    #Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)
    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)


cap.release()
output_file.release()
cv2.destroyAllWindows()
