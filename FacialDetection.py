import numpy as np
import cv2

pathToResource = "./Resources/"

# ===== Resources Loading =====
# Videos
videosToAdd = [
    "street.mp4",
    "exercise.mp4",
    "office.mp4"
]
talkingVideo = cv2.VideoCapture(pathToResource + "talking.mp4")
talkingVideoFrame = int(talkingVideo.get(cv2.CAP_PROP_FRAME_COUNT))

# Watermarks
watermark1 = cv2.imread(pathToResource + "watermark1.png")
watermark2 = cv2.imread(pathToResource + "watermark2.png")

# Face Haar Cascade model loading
face_cascade = cv2.CascadeClassifier("face_detector.xml")
# ===== Resources Loading ===== 

# resolution is 1280x720
dimension = (1280, 720) 

# universal count to keep track of talking video reset
count = 0

# assuming watermark width and height are the same size as backgroundFrame
def placeWatermark(backgroundFrame, watermark):
    waterHeight, waterWidth, c = watermark.shape # row, col, channels
    watermarkGray = cv2.cvtColor(watermark,cv2.COLOR_BGR2GRAY) # Convert watermark image to grayscale
    
    # Creating the mask of the watermarks as well as applying an inverse to the mask
    ret, mask = cv2.threshold(watermarkGray, 10, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.bitwise_and(watermark, watermark, mask) # Take only the region of watermark from watermark image
    output = cv2.add(backgroundFrame, mask) # Combine extracted watermark image to background image

    backgroundFrame[0:waterHeight, 0:waterWidth] = output # Set the ROI of the background image to the watermark

if __name__ == "__main__":
    for video in videosToAdd:
        
        # output video is 30FPS, 1280x720
        outputVideo = cv2.VideoWriter('processed_' + video + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, dimension)
        
        # Get the background video to add so we can load in the information needed
        backgroundVideo = cv2.VideoCapture(pathToResource + video)

        # Get the background video's total frames
        backgroundVidTotalFrame = int(backgroundVideo.get(cv2.CAP_PROP_FRAME_COUNT))

        # Go through and process all the frames in the video
        for currentFrameCount in range(0, backgroundVidTotalFrame):

            if(count % talkingVideoFrame == 0):
                # reload talking video
                talkingVideo = cv2.VideoCapture(pathToResource + "talking.mp4")

            backgroundSuccess, backgroundFrame = backgroundVideo.read()
            talkignSuccess, talkingFrame = talkingVideo.read()

            backgroundFrame = cv2.resize(backgroundFrame, dimension, interpolation = cv2.INTER_AREA)

            gray = cv2.cvtColor(backgroundFrame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # ===== Blur all faces in backgroundFrame =====
            for (top, right, bottom, left) in faces:           
                face = backgroundFrame[right:right+left, top:top+bottom]
                face = cv2.GaussianBlur(face,(23, 23), 30)
                backgroundFrame[right:right+face.shape[0], top:top+face.shape[1]] = face
            # ===== Blur all faces in backgroundFrame =====

            # ===== Insert talking video =====

            # resize talking video talkingFrame 
            talkingFrame = cv2.resize(talkingFrame, (300, 180)) # Resize the talking video to a smaller size
            talkingWidth, talkingHeight, talkingChannel = talkingFrame.shape # Take the shape of the resized video

            backgroundFrame[100:talkingWidth+100,100:talkingHeight+100,0:talkingChannel] = 0 # Create a empty area to perform the task
            
            regionOfInterest = backgroundFrame[100:talkingWidth+100,100:talkingHeight+100] # Create the ROI of the background video
            backgroundFrame = cv2.rectangle(backgroundFrame,(100,100),(400,280),(0,0,0),20) # Create the border for the talking video
            
            talkingGray = cv2.cvtColor(talkingFrame,cv2.COLOR_BGR2GRAY) # Convert the talking video frames colors to grayscale
            
            # Creating the mask of the talking video
            ret, talkingMask = cv2.threshold(talkingGray, 0, 0, cv2.THRESH_BINARY) 

            maskInv = cv2.bitwise_not(talkingMask) # Create an inverse mask of the talking video

            # Black-out the area of the background video in ROI
            img1 = cv2.bitwise_and(regionOfInterest,regionOfInterest,maskInv)
            
            # Take the region of talking video that its mask requires
            img2 = cv2.bitwise_and(talkingFrame,talkingFrame,talkingMask)
            
            # Combine both the blacked-out area and the talking video ROI togther to create the final output video
            outputImg = img1 + img2 
            
            # Set the background video area to the output video
            backgroundFrame[100:talkingWidth+100,100:talkingHeight+100] = outputImg
            # ===== Insert talking video =====
            
            # ===== Insert watermark =====
            if currentFrameCount <= 80:
                placeWatermark(backgroundFrame, watermark1)
            
            else:
                placeWatermark(backgroundFrame, watermark2)
            # ===== Insert watermark =====

            count += 1
            print(f"Frame processing {currentFrameCount+1}/{backgroundVidTotalFrame} for {video}", end='\r')
            cv2.waitKey(10) # wait 50 miliseconds
            outputVideo.write(backgroundFrame)
        
        print("\n")
        outputVideo.release()
        backgroundVideo.release()

cv2.waitKey()
cv2.destroyAllWindows()