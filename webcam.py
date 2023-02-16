import numpy as np
import cv2 as cv
import imageio


mtx = np.load('mtx.npy')
dist = np.load('dist.npy')


# BGR
def draw(img, corners, imgpts, ret):
    cv.drawChessboardCorners(img, (9, 6), corners2, ret)

    # 4 bottom points of the cube
    corner = tuple(corners[0].ravel())
    corner1 = tuple(corners[1].ravel())
    corner11 = tuple(corners[10].ravel())
    corner10 = tuple(corners[9].ravel())

    # imgpoints
    imgpts0 = tuple(imgpts[0].ravel())
    imgpts1 = tuple(imgpts[1].ravel())
    imgpts2 = tuple(imgpts[2].ravel())

    # point5 = [int((corner[0]-imgpts0[0])*2/3+imgpts0[0]), int((corner[1]-imgpts0[1])*2/3+imgpts0[1])]
    # point6 = [int((corner[0]-imgpts1[0])*2/3+imgpts1[0]), int((corner[1]-imgpts1[1])*2/3+imgpts1[1])]
    # point7 = [int(point5[0]-corner[0]+point6[0]), int(point5[1]-corner[1]+point6[1])]

    # upper points of the cube
    point1 = [int((corner[0] - imgpts2[0]) * 2 / 3 + imgpts2[0]), int((corner[1] - imgpts2[1]) * 2 / 3 + imgpts2[1])]
    point2 = [int((corner1[0] - (corner[0] - imgpts2[0]) * 1 / 3)), int(corner1[1] - (corner[1] - imgpts2[1]) * 1 / 3)]
    point3 = [int((corner11[0] - (corner[0] - imgpts2[0]) * 1 / 3)),
              int(corner11[1] - (corner[1] - imgpts2[1]) * 1 / 3)]
    point4 = [int((corner10[0] - (corner[0] - imgpts2[0]) * 1 / 3)),
              int(corner10[1] - (corner[1] - imgpts2[1]) * 1 / 3)]

    # Draw the 4 vertical sides of the cube
    img = cv.line(img, (int(corner[0]), int(corner[1])), (point1[0], point1[1]), (255, 255, 0), 4)
    img = cv.line(img, (int(corner1[0]), int(corner1[1])), (point2[0], point2[1]), (255, 255, 0), 4)
    img = cv.line(img, (int(corner11[0]), int(corner11[1])), (point3[0], point3[1]), (255, 255, 0), 4)
    img = cv.line(img, (int(corner10[0]), int(corner10[1])), (point4[0], point4[1]), (255, 255, 0), 4)

    # Draw upper
    img = cv.line(img, (point1[0], point1[1]), (point4[0], point4[1]), (255, 255, 0), 4)
    img = cv.line(img, (point1[0], point1[1]), (point2[0], point2[1]), (255, 255, 0), 4)
    img = cv.line(img, (point2[0], point2[1]), (point3[0], point3[1]), (255, 255, 0), 4)
    img = cv.line(img, (point3[0], point3[1]), (point4[0], point4[1]), (255, 255, 0), 4)

    '''
    img = cv.line(img, (int(corner[0]),int(corner[1])) , (point5[0],point5[1]),(255, 255, 0), 4)
    img = cv.line(img, (int(corner[0]),int(corner[1])) , (point6[0],point6[1]),(255, 255, 0), 4)
    img = cv.line(img, (point5[0],point5[1]) , (point7[0],point7[1]),(255, 255, 0), 4)
    img = cv.line(img, (point6[0],point6[1]) , (point7[0],point7[1]),(255, 255, 0), 4)
    '''

    # Draw bottom
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(corner1[0]), int(corner1[1])), (255, 255, 0), 4)
    img = cv.line(img, (int(corner1[0]), int(corner1[1])), (int(corner11[0]), int(corner11[1])), (255, 255, 0), 4)
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(corner10[0]), int(corner10[1])), (255, 255, 0), 4)
    img = cv.line(img, (int(corner11[0]), int(corner11[1])), (int(corner10[0]), int(corner10[1])), (255, 255, 0), 4)

    # y axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts0[0]), int(imgpts0[1])), (255, 0, 0), 10)
    # x axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts1[0]), int(imgpts1[1])), (0, 255, 0), 10)
    # z axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts2[0]), int(imgpts2[1])), (0, 0, 255), 10)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
axis2 = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

cap = cv.VideoCapture(0)
imgs = []
while True:
    ret, img = cap.read()
    if ret:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # print(imgpts)
        img = draw(img, corners2, imgpts, ret)
        # img = drawbox(img, corners2, imgpts)
        # img = cv.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
    cv.imshow('webcam', img)
    imgs.append(img)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
if len(imgs):
    imageio.mimsave('webcam.gif', imgs, 'GIF', duration=0.1)
cap.release()
cv.destroyAllWindows()