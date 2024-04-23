import cv2
import numpy as np
 
image = cv2.imread("rustic.jpeg")

#image3=image.copy()
#pts1 = np.float32([(150, 127), (750, 127), (80, 510), (815, 510)])
#pts2 = np.float32([(0, 0), (800, 0), (0, 600), (800, 600)])   
#matrix = cv2.getPerspectiveTransform(pts1, pts2)
#image = cv2.warpPerspective(image3, matrix, (800, 600))
default = image.copy()


image2= image.copy()

def trackbarcallback(value):
    #print(value)
    pass
    

cv2.namedWindow('trackbar')
cv2.createTrackbar('par1','trackbar',90,100,trackbarcallback)
cv2.createTrackbar('par2','trackbar',60,100,trackbarcallback)


def trackbar_values():
    return cv2.getTrackbarPos('par1','trackbar'),cv2.getTrackbarPos('par2','trackbar')


while cv2.waitKey(1) != ord('q'):
    par_1 ,par_2= trackbar_values()
    image=default.copy()
    image2=default.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        
    canny_image=image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray_image, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1 = par_1, param2 = par_2, minRadius = 0, maxRadius = 100)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]

                #cv2.circle(image2, (x, y), r, (0, 255, 0), 2)
                #cv2.circle(image2, (x, y), 1, (0, 0, 255), 3)
            x1, y1 = x - r, y - r
            x2, y2 = x + r, y + r
            cv2.rectangle(image2, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print(len(detected_circles[0,:]))
    cv2.imshow("Detected Circle", image2) 
    #cv2.imshow("canny",canny_image)



cv2.waitKey(0)
cv2.destroyAllWindows() 
