import cv2
import numpy as np
import math
 
cap = cv2.VideoCapture("VTemplate.mp4")
ret, frame = cap.read()
background = frame
points = []
ready = False
first = True

x1, y1 = 0,0
x2, y2 = 0,0

backSub = cv2.createBackgroundSubtractorMOG2()

def trackbarcallback(value):
    #print(value)
    pass

cv2.namedWindow('trackbar')
cv2.createTrackbar('par1','trackbar',70,100,trackbarcallback)
cv2.createTrackbar('par2','trackbar',60,100,trackbarcallback)
cv2.createTrackbar('thresh','trackbar',150,500,trackbarcallback)



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

    _, thresh = cv2.threshold(resized_img, threshold, 255, cv2.THRESH_BINARY)           # Threshold para identificar pixeis diferentes de branco
    cv2.imwrite(f"Thresh.png",thresh)
    black_pixels = thresh.size - cv2.countNonZero(thresh)                         # Conta numero de pixeis pretos
    #print(black_pixels)
    if(black_pixels > 20000):
        return True
    else:
        return False

def text_feedback(feedback,color):
    new_x1, new_y1 = x1, y1 - 30
    new_x2, new_y2 = x2, y1
    text = feedback
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = new_x1 + (new_x2 - new_x1 - text_size[0]) // 2
    text_y = new_y1 + (new_y2 - new_y1 + text_size[1]) // 2
    cv2.rectangle(frame2, (new_x1, new_y1), (new_x2, new_y2), color, -1)
    cv2.putText(frame2, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f'Ponto selecionado: {x}, {y}')


def detect_fork(contour):
    #cv2.drawContours(frame3, contours,-1, (0, 255, 0), 2)
    fork_template = cv2.imread("fork_template.png")
    gray_fork = cv2.cvtColor(fork_template, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_fork, 127, 255, cv2.THRESH_BINARY)
    moments_template = cv2.moments(thresh)
    moments_template = cv2.moments(gray_fork)
    hu_moments_template = cv2.HuMoments(moments_template)


    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments)

    for i in range(0, 7):
        hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
        hu_moments_template[i] = -np.sign(hu_moments_template[i]) * np.log10(abs(hu_moments_template[i]))

    distancia = np.linalg.norm(hu_moments_template - hu_moments)
    print(distancia)

    if(distancia < 10):
        new_x1, new_y1 = x, y - 30
        new_x2, new_y2 = x+w, y
        text = "fork"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = new_x1 + (new_x2 - new_x1 - text_size[0]) // 2
        text_y = new_y1 + (new_y2 - new_y1 + text_size[1]) // 2
        cv2.rectangle(frame3, (new_x1, new_y1), (new_x2, new_y2), (0,0,255), -1)
        cv2.putText(frame3, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    #similarity = abs(hu_moments_template-hu_moments)
    #print(similarity)

def detect_knife(contour):

    #cv2.drawContours(frame3, contours,-1, (0, 255, 0), 2)
    knife_template = cv2.imread("knife_template.png")
    
    gray_knife = cv2.cvtColor(knife_template, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_knife, 127, 255, cv2.THRESH_BINARY)
    moments_template = cv2.moments(thresh)
    hu_moments_template = cv2.HuMoments(moments_template)


    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments)

#-------------------Algoritmo Copiado--------------------------------------------
    for i in range(0, 7):
        hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
        hu_moments_template[i] = -np.sign(hu_moments_template[i]) * np.log10(abs(hu_moments_template[i]))

    distancia = np.linalg.norm(hu_moments_template - hu_moments)
#---------------------------------------------------------------------------------
    print(distancia)

    if(distancia < 20):
        new_x1, new_y1 = x, y - 30
        new_x2, new_y2 = x+w, y
        text = "Knife"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = new_x1 + (new_x2 - new_x1 - text_size[0]) // 2
        text_y = new_y1 + (new_y2 - new_y1 + text_size[1]) // 2
        cv2.rectangle(frame3, (new_x1, new_y1), (new_x2, new_y2), (0,0,255), -1)
        cv2.putText(frame3, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)



        #text_feedback("Faca",(0, 0, 255))
    #similarity = abs(hu_moments_template-hu_moments)
    #print(similarity)
    #if(similarity > 50):
    #    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)


 

def text_amount(dirty_plates,messy_objects):
    width, height = 500, 200
    black_bg = np.zeros((height, width, 3), dtype=np.uint8)  

    text1 = "Amount of dirty plates: " + str(dirty_plates)
    text2 = "Amount of messy objects: " + str(messy_objects)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    thickness = 2
    (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    x1 = (width - text_width1) // 2
    y1 = height // 3

    x2 = (width - text_width2) // 2
    y2 = 2 * height // 3
    cv2.putText(black_bg, text1, (x1, y1), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(black_bg, text2, (x2, y2), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Quantity', black_bg)

def centers_distance(x1,y1,x2,y2):
    if x1 <= x2:
        width = x2 - x1
    else:
        width = x1 - x2
                        
    if y1 <= y2:
        height = y2 - y1
    else:
        height = y1 - y2
                
    return math.sqrt((width ** 2) + (height ** 2))

def detect_overlapping(detected_circles):
    dirty_plates=0
    first_cycle=True
    if((len(detected_circles[0,:]))>=2):
        #print(len(detected_circles[0,:]))
        for i in range(len(detected_circles[0,:])):
            for j in range(i+1,len(detected_circles[0,:])):
                x1, y1, r1 = detected_circles[0, i]  
                x2, y2, r2 = detected_circles[0, j]
                distance = centers_distance(x1,y1,x2,y2)
                if distance < r1 + r2:
                    if(first_cycle):
                        dirty_plates+=1
                        first_cycle=False
                    print(f"The plate {i} and plate {j} are overlapping")
                    dirty_plates+=1
    return dirty_plates

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

while cv2.waitKey(33) != ord('q'):                           
    par_1 ,par_2,threshold= trackbar_values()
    dirty_plates=0
    messy_objects = 0
    frame4= frame.copy()


    if len(points) < 2:
        cv2.imshow('Frame', frame)
    elif (ready is False):

        top_left = points[0]
        bottom_right = points[1]
        cv2.rectangle(frame4, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow("ROI", frame4)
        cv2.destroyWindow('Frame')
        ready = True


    

    if(ready):
        ret, frame = cap.read()  
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        frame = frame[y_min:y_max, x_min:x_max]
        frame2= frame.copy()
        frame3= frame.copy()
        #cv2.imshow("frame3",frame)

        if(first):
            first = False
            background = frame
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(blurred_image, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        detected_circles = cv2.HoughCircles(dilation, cv2.HOUGH_GRADIENT, 1, 50, param1 = par_1, param2 = par_2, minRadius = 0, maxRadius = 1000)    

        key = cv2.waitKey(1)
        if key == ord('s'):
            background = frame.copy()
        
        sub = cv2.subtract(frame,background)
        cv2.imshow("sub",sub)
        gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_sub, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        messy_objects=len(contours)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame3, (x, y), (x + w, y + h), (0, 0, 255), 2)
                detect_knife(contour)
                #detect_fork(contour)
            else: 
                messy_objects-=1

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                x, y, r = pt[0], pt[1], pt[2]
                x1, y1 = x - r, y - r
                x2, y2 = x + r, y + r
                is_dirty = check_dirty(frame,pt,x1,y1,x2,y2,threshold)
                if is_dirty:
                    dirty_plates+=1
                    text_feedback("Dirty",(255, 0, 0))
                else:
                    text_feedback("Clean",(255, 0, 0))
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)

            dirty_plates+=detect_overlapping(detected_circles)
        
        text_amount(dirty_plates,messy_objects)


        cv2.imshow("Detected Plates", frame2)
        cv2.imshow("Detected messy dishes",frame3)
        messy_plates=0





cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows() 
