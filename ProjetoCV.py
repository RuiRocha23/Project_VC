import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)

def trackbarcallback(value):
    #print(value)
    pass
    

cv2.namedWindow('trackbar')
cv2.createTrackbar('par1','trackbar',90,100,trackbarcallback)
cv2.createTrackbar('par2','trackbar',60,100,trackbarcallback)
cv2.createTrackbar('thresh','trackbar',1,500,trackbarcallback)


def trackbar_values():
    return cv2.getTrackbarPos('par1','trackbar'),cv2.getTrackbarPos('par2','trackbar'),cv2.getTrackbarPos('thresh','trackbar')

def check_dirty(image, circle,x1,y1,x2,y2,threshold):
    x, y, r = circle[0], circle[1], circle[2]
    circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)             # Extrai a região dentro do circulo
    cv2.circle(circle_mask, (x, y), r, 255, -1)
    circle_roi = cv2.bitwise_and(image, image, mask=circle_mask)        #Região de interesse
    circle_gray = cv2.cvtColor(circle_roi, cv2.COLOR_BGR2GRAY)          # Converte para grayscale
    
    x1,y1 = max(0, x1), max(0, y1)
    x2,y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    cropped_image = circle_gray[y1:y2, x1:x2]

    height, width = cropped_image.shape[:2]

    if height > width:
        scale_factor = 250 / height
    else:
        scale_factor = 250 / width

    resized_img = cv2.resize(cropped_image, (int(width * scale_factor), int(height * scale_factor)))


    cv2.imwrite(f"Captura.png",resized_img)
    #thresh=cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, (threshold*2+1), c)
    _, thresh = cv2.threshold(resized_img, threshold, 255, cv2.THRESH_BINARY) # Threshold para identificar pixeis diferentes de branco
    cv2.imwrite(f"Thresh.png",thresh)
    non_white_pixels = thresh.size - cv2.countNonZero(thresh)                         # Conta numero de pixeis pretos
    print(non_white_pixels)
    if(non_white_pixels > 20000):
        return True
    else:
        return False



while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    ret, frame2= cap.read()
    par_1 ,par_2,threshold= trackbar_values()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    detected_circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, 50, param1 = par_1, param2 = par_2, minRadius = 0, maxRadius = 100)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]
            x1, y1 = x - r, y - r
            x2, y2 = x + r, y + r
            is_dirty = check_dirty(frame,pt,x1,y1,x2,y2,threshold)
            if is_dirty:
                new_x1, new_y1 = x1, y1 - 30
                new_x2, new_y2 = x2, y1
                text = "Dirty"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = new_x1 + (new_x2 - new_x1 - text_size[0]) // 2
                text_y = new_y1 + (new_y2 - new_y1 + text_size[1]) // 2
                cv2.rectangle(frame2, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 0), -1)
                cv2.putText(frame2, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            else:
                new_x1, new_y1 = x1, y1 - 30
                new_x2, new_y2 = x2, y1
                text = "Clean"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = new_x1 + (new_x2 - new_x1 - text_size[0]) // 2
                text_y = new_y1 + (new_y2 - new_y1 + text_size[1]) // 2
                cv2.rectangle(frame2, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 0), -1)
                cv2.putText(frame2, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

            #print(is_dirty)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print(len(detected_circles[0,:]))

    cv2.imshow("Detected Circle", frame2) 
    #cv2.imshow("canny",canny_image)


cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows() 
