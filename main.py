import cv2
import numpy as np
import utils
from PIL import Image

#################################
path = ["1.jpeg", "2.jpeg", "3.jpeg", "4.jpeg", "5.jpeg", "6.jpeg"]
widthImg = 960
heightImg = 630
widthWarp = 290
heightWarp = 650
questions = 20
choices = 5
answers = open("answers.txt", "r")
#################################

img = cv2.imread(path[5])
img = cv2.resize(img, (widthImg, heightImg))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)
imgCanny = cv2.Canny(imgBlur, 20, 50)
imgContours = img.copy()
imgBigContour = img.copy()
imgFinal = img.copy()

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

rectCon = utils.rectContour(contours)

biggestPointsList = []

for rectangle in rectCon:
    biggestPointsList.append(utils.getCornerPoints(rectangle))

for i in range(len(biggestPointsList)):
    biggestPointsList[i] = utils.reorder(biggestPointsList[i])

biggestPoints1 = biggestPointsList[0]

answer = []

for i in answers:
    k = []
    for j in i:
        if j != '\n':
            k.append(int(j))
    answer.append(k)

ans = answer[0]

if biggestPoints1.size != 0:
    #biggestPoints1 = utils.reorder(biggestPoints1)
    cv2.drawContours(imgBigContour, biggestPoints1, -1, (0, 255, 0), 20)
    pts1 = np.float32(biggestPoints1)
    pts2 = np.float32([[0,0], [widthWarp, 0], [0, heightWarp], [widthWarp, heightWarp]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthWarp, heightWarp))

    imgWarpOriginal = imgWarpColored.copy()

    imgWarpColored = imgWarpColored[50:heightWarp,50:widthWarp]

    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 190, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThresh, questions, choices)
    myPixelVal = np.zeros((questions, choices))

    for x in range(questions):
        for y in range(choices):
            myPixelVal[x][y] = cv2.countNonZero(boxes[x][y])

    myIndex = []

    for x in range(questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])

    grading = []
    for i in range(questions):
        if myIndex[i] == ans[i]:
            grading.append(1)
        else:
            grading.append(0)
    score = sum(grading)/questions * 100

    utils.showAnswers(imgWarpColored, myIndex, grading, ans, questions, choices)
    imgRawDrawings = np.zeros_like(imgWarpColored)
    imgRawDrawingsOr = np.zeros_like(imgWarpOriginal)

    for i in range(50,heightWarp):
        for j in range(50, widthWarp):
            imgWarpOriginal[i][j] = imgWarpColored[i-50][j-50]

    utils.showAnswers(imgRawDrawings, myIndex, grading, ans, questions, choices)

    for i in range(50,heightWarp):
        for j in range(50, widthWarp):
            imgRawDrawingsOr[i][j] = imgRawDrawings[i-50][j-50]

    invMatrix = cv2.getPerspectiveTransform(pts2, pts1)  # INVERSE TRANSFORMATION MATRIX
    imgInvWarp = cv2.warpPerspective(imgRawDrawingsOr, invMatrix, (widthImg, heightImg))  # INV IMAGE WARP

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)


cv2.imshow("original", imgFinal)
cv2.waitKey(0)

cv2.imwrite("FinalImg.jpeg", imgFinal)