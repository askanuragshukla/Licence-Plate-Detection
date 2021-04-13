import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

states = {"AN" :"ANDAMAN AND NICOBAR","AP":"ANDHRA PRADESH", "AR":"ARUNACHAL PRADESH","	AS":"ASSAM","BR":"BIHAR", "CH":"CHANDIGARH","DN":"DADRA AND NAGAR HAVELI",
           "DD":"DAMAN AND DIU", "DL":"DELHI", "GA":"GOA", "GJ":"GUJRAT", "HR":"HARYANA", "HP":"HIMACHAL PRADESH", "JK":"JAMMU AND KASHMIR", "KA":"KARNATAKA", "KL":"KERLA",
           "LD":"LAKSHADWEEP", "MP":"MADHYA PRADESH", "MH":"MAHARASTRA", "MN":"MANIPUR","ML":"MEGHALAYA", "MZ":"MIZORAM", "NL":"NAGALAND", "OR":"ORRISA","PY":"PONDICHERRY",
           "PN":"PUNJAB", "RJ":"RAJASTHAN", "SK":"SIKKIM", "TN":"TAMILNADU", "TR":"TRIPURA", "UP":"UTTARPRADESH", "WB":"WESTBENGAL"}

img = cv2.imread('img1.jpg',cv2.IMREAD_COLOR)
img = imutils.resize(img, width=500 )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1=img.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)
# cv2.imshow("img1",img1)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
screenCnt = None #will store the number plate contour
img2 = img.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
# cv2.imshow("img2",img2) #top 30 contours

count=0
idx=7
# loop over contours
for c in cnts:
  # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4: #chooses contours with 4 corners
                screenCnt = approx
                x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
                new_img=img[y:y+h,x:x+w]
                cv2.imwrite('./'+str(idx)+'.png',new_img) #stores the new image
                idx+=1
                break
            #draws the selected contour on original image


Cropped_loc='./7.png' #the filename of cropped image
cv2.imshow("cropped",cv2.imread(Cropped_loc))
pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract.exe' #exe file for using ocr

text=pytesseract.image_to_string(Cropped_loc, lang='eng') #converts image characters to string
read = ''.join(e for e in text if e.isalnum())
stat = read[0:2]
print('This car belongs to : ', states[stat])
print('number is ',read)
cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 5)
cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
cv2.imshow('plate', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import pytesseract
# import numpy as np
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# cascade = cv2.CascadeClassifier("haarcascade_russian_platte_number.xml")
#
# states = {"AN" :"ANDAMAN AND NICOBAR","AP":"ANDHRA PRADESH", "AR":"ARUNACHAL PRADESH","	AS":"ASSAM","BR":"BIHAR", "CH":"CHANDIGARH","DN":"DADRA AND NAGAR HAVELI",
#           "DD":"DAMAN AND DIU", "DL":"DELHI", "GA":"GOA", "GJ":"GUJRAT", "HR":"HARYANA", "HP":"HIMACHAL PRADESH", "JK":"JAMMU AND KASHMIR", "KA":"KARNATAKA", "KL":"KERLA",
#           "LD":"LAKSHADWEEP", "MP":"MADHYA PRADESH", "MH":"MAHARASTRA", "MN":"MANIPUR","ML":"MEGHALAYA", "MZ":"MIZORAM", "NL":"NAGALAND", "OR":"ORRISA","PY":"PONDICHERRY",
#           "PN":"PUNJAB", "RJ":"RAJASTHAN", "SK":"SIKKIM", "TN":"TAMILNADU", "TR":"TRIPURA", "UP":"UTTARPRADESH", "WB":"WESTBENGAL"}
#
# def extract_img(img_name):
#     global read
#     img = cv2.imread(img_name)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     nplate = cascade.detectMultiScale(gray,1.1,4)
#     for (x,y,w,h) in nplate:
#         a,b = (int(0.02*img.shape[0], int(0.02*img.shape[1]))
#
#
#
#         kernel = np.ones((1, 1), np.uint8)
#         plate = cv2.dilate(img[y+a:y+h-a, x+b:x+w-b, :], kernel, iterations=1)
#         plate = cv2.erode(img[y+a:y+h-a, x+b:x+w-b, :], kernel, iterations=1)
#         palte_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#         (thres, plate) = cv2.threshold(palte_gray, 127, 255, cv2.THRESH_BINARY)
#
#         read = pytesseract.image_to_string(plate)
#
#         read = ''.join(e for e in read if e.isalnum())
#
#         stat = read[0:2]
#
#         try:
#             print('This car belongs to : '
#             states[stat])
#             except:
#             print('OOPS unknown State')
#         print(read)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
#         cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
#         cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
#         cv2.imshow('plate', plate)
#     cv2.imshow("result", img)
#     cv2.imwrite("result.jpg", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# extract_img('licenceplate.py')
#
#
#
#
#
#
#
#
#
#
#
