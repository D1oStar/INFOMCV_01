import numpy as np
import cv2 as cv
import glob

click = 0
manual_position = np.zeros((4, 2), np.float32)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

#mouse click event
def click_event(event, x, y, flags, params):
    global click
    global manual_position
    if event == cv.EVENT_LBUTTONDOWN:
        if click < 4:
            manual_position[click] = (x, y)
            # print(manual_position)
            cv.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv.imshow('img', img)
            click += 1


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#points for rejecting input images with low quality 
imgpoints2 = [] 
objpoints2 = [] 

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('*.jpg')
h, w = 0, 0

for fname in images:

    img = cv.imread(fname)
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret:

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints2.append(objp)
        imgpoints2.append(corners2) 
        #rejection of low quality input images 
        rms2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(objpoints2, imgpoints2, (h, w), None, None)
        imgpoints3, _ = cv.projectPoints(objpoints2[0], rvecs2[0], tvecs2[0], mtx2, dist2)
        error = cv.norm(imgpoints2[0], imgpoints3, cv.NORM_L2)/len(imgpoints2)
        imgpoints2 = []
        objpoints2 = []
        print(error)
        
        if error < 10:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)       
            imgpoints.append(corners2)
            #cv.drawChessboardCorners(img, (9, 6), corners2, ret)
            print(fname)

    else:
        #Manual marking        
        print('Requires manual marking of 4 points')
        img = cv.imread(fname)
        img = cv.resize(img, (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
        h, w = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)        
        click = 0
        manual_position = np.zeros((4, 2), np.float32)
        cv.imshow('img', img)
        cv.setMouseCallback('img', click_event)
        cv.waitKey(0)
        if click == 4:
            print(manual_position)
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            polygon = np.array(manual_position, np.int32)
            cv.fillConvexPoly(mask, polygon, (255))
            imgc = gray.copy()            
            imgc = imgc * mask
            imgc = 255 - imgc
            ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

            if ret:
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints2.append(objp)
                imgpoints2.append(corners2) 
                #rejection of low quality input images 
                rms2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(objpoints2, imgpoints2, (h, w), None, None)
                imgpoints3, _ = cv.projectPoints(objpoints2[0], rvecs2[0], tvecs2[0], mtx2, dist2)
                error = cv.norm(imgpoints2[0], imgpoints3, cv.NORM_L2)/len(imgpoints2)
                imgpoints2 = []
                objpoints2 = []
                print(error)
        
                if error < 10:

                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, (9, 6), corners2, ret)
                    print(fname)
                    cv.imshow('img', img)
                    cv.waitKey(0)
                    # print(corners)
            else:
                #Remove shadows
                # Calculate the mean value of the grey and white pixels
                pixel = int(np.mean(img[img > 140]))
                # Change the off-white part to a colour close to the background
                img[img > 60] = pixel

                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                
                ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

                if ret:

                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    objpoints2.append(objp)
                    imgpoints2.append(corners2) 
                    #rejection of low quality input images 
                    rms2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(objpoints2, imgpoints2, (h, w), None, None)
                    imgpoints3, _ = cv.projectPoints(objpoints2[0], rvecs2[0], tvecs2[0], mtx2, dist2)
                    error = cv.norm(imgpoints2[0], imgpoints3, cv.NORM_L2)/len(imgpoints2)
                    imgpoints2 = []
                    objpoints2 = []
                    print(error)
                    
                    if error < 10:
                        objpoints.append(objp)
                        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)       
                        imgpoints.append(corners2)
                        print(fname)
                else:
                    print("fail to find in %s" % fname)


rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (h, w), None, None)


outfile1 = 'mtx'
np.save(outfile1, mtx)
outfile2 = 'dist'
np.save(outfile2, dist)
