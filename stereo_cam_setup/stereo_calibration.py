import numpy as np
import cv2 as cv
import glob

chessboardSize = (9, 6)
frameSize = (640, 480)

criterea=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp=np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
objp[:,:2]=np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objpoints=[]
imgpointsL=[]
imgpointsR=[]
imagesLeft=glob.glob('images/stereoLeft/*.png')
imagesRight=glob.glob('images/stereoRight/*.png')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL=cv.imread(imgLeft)
    imgR=cv.imread(imgRight)
    grayL=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR=cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL=cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR=cv.findChessboardCorners(grayR, chessboardSize, None)

    if retL==True and retR==True:
        objpoints.append(objp)
        corners2L=cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criterea)
        corners2R=cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criterea)
        imgpointsL.append(corners2L)
        imgpointsR.append(corners2R)

        cv.drawChessboardCorners(imgL, chessboardSize, corners2L, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, corners2R, retR)
        cv.imshow('imgL', imgL)
        cv.imshow('imgR', imgR)
        cv.waitKey(100)
cv.destroyAllWindows()

# Stereo Calibration
retL, cameraMatrixL, distL, rvecsL, tvecsL=cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL=imgL.shape
newCameraMatrixL, roi_L=cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR=cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR=imgR.shape
newCameraMatrixR, roi_R=cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

flag=0
flag|=cv.CALIB_FIX_INTRINSIC
criterea_stereo=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStoreo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix=cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criterea_stereo, flag)


rectifyScale=1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R=cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL=cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR=cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Done Calibration")
cv_file=cv.FileStorage("stereoCalibration.xml", cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

cv_file.release()