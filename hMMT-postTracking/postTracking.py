# -*- coding: utf-8 -*-
"""
postTracking
@author: Haben Y. Abraha
"""
from tkinter import filedialog
import cv2
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class ClickLocation:
        
    def __init__(self):
        self.roi_center = None
        
    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_center = (x,y)
            
      
vid_path = filedialog.askopenfilename(
        initialdir="C:/",
        filetypes=(("All Files","*.*"), ("Text File", "*.txt")), 
        title= "Choose a file.")

contraction_vid = cv2.VideoCapture(vid_path)
colour = (255,50,50)
frame_counter = 1
user_roi = False
gray_history = []
fx = 0.55
fy= 0.55


ok, first_frame = contraction_vid.read()
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.equalizeHist(first_frame_gray)

if not ok:
    print("Cannot read first frame")
    sys.exit()

'''
Due to screen size constraints, images shown to user are all resized versions 
of the frames that the tracking is done on. ROI format is:
(top_left_corner_x_coord, top_left_corner_y_coord, ROI_width, ROI_height).
'''
#ROI width/height factors. Adjust if necessary.

#larger ROI - less sensitive but better able to track very large deflections
#Smaller ROI - more sensitive, better able to track smaller deflections

#RHF = 0.25
#RWF = 0.08

#RHF = 0.2
#RWF = 0.06

#Default ROI size
RHF = 0.1
RWF = 0.03

resized_frame = cv2.resize(first_frame, (0,0), fx=fx, fy=fy)  
     
cv2.namedWindow("Identify post")
clickLocation = ClickLocation()
cv2.setMouseCallback("Identify post", clickLocation.click)

while (True):
    
    marked_frame = resized_frame.copy()    
    if not clickLocation.roi_center is None:
        cv2.circle(marked_frame,(clickLocation.roi_center[0],
                                 clickLocation.roi_center[1]), 5,(255,0,0),1)
        
        #Get user click location, use to draw ROI 
        roi_center = clickLocation.roi_center 

        user_top_left = (int(roi_center[0] - 
                             round(RWF * resized_frame.shape[1]/2)), 
        int(roi_center[1] - round(RHF * resized_frame.shape[0]/2)))

        user_bottom_right = (int(roi_center[0] + 
                                 round(RWF * resized_frame.shape[1]/2)), 
        int(roi_center[1] + round(RHF * resized_frame.shape[0]/2)))

        user_roi = (user_top_left[0], user_top_left[1], 
                    user_bottom_right[0] - user_top_left[0],
                    user_bottom_right[1] - user_top_left[1])
        
        cv2.rectangle(marked_frame, (user_top_left), (user_bottom_right),
                      colour, 1)    
    cv2.imshow("Identify post", marked_frame) 

    h = cv2.waitKey(1)
    if h & 0xFF == ord('\r'):
        break

roi = (round(user_roi[0]/fx), round(user_roi[1]/fy),
       round(user_roi[2]/fx), round(user_roi[3]/fy))

top_left = (roi[0], roi[1])
bottom_right = (roi[0] + roi[2], roi[1] + roi[3])

#Draw rectangle. Rectangle takes top left and bottom right points as params
cv2.rectangle(resized_frame, (user_top_left), (user_bottom_right), colour, 1)
cv2.imshow("Frame", resized_frame)
cv2.moveWindow("Frame", 0,0)

cv2.waitKey(0)
cv2.destroyWindow("Frame")

tracker = cv2.TrackerKCF_create()

#Initialize tracker with first frame
tracker.init(first_frame_gray, roi)
gray_history.append(first_frame_gray)

#log x-axis location of post, i.e. center of roi
post_location = [roi[0] + roi[2]/2]

#Go to second frame
unfinished, frame = contraction_vid.read()

while unfinished:

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    resized_frame = cv2.resize(frame, (0,0), fx=fx, fy=fy)
    tracked, roi = tracker.update(frame_gray)
    gray_history.append(frame_gray)
    
    if tracked:
        #Tracker has identified post
        
        #Roi points are floating point, must cast to int

        top_left = (int(roi[0]), int(roi[1]))
        bottom_right = (int(roi[0]) + int(roi[2]), int(roi[1]) + int(roi[3]))
        
        resized_top_left = (round(top_left[0]*fx),round(top_left[1]*fy))
        resized_bottom_right =  (resized_top_left[0] + round(roi[2]*fx),
                                 resized_top_left[1] + round(roi[3]*fy))
        
        #Resize roi for depicting w/ imshow to fit screen
        cv2.rectangle(resized_frame, resized_top_left,resized_bottom_right,
                      colour, 1)
        
        cv2.imshow("Frame",resized_frame)        
        cv2.waitKey(1)
        
        #log x-axis location of post
        post_location.append(top_left[0] + roi[2]/2)
    else:
        
        #If tracking failed, roi becomes = (0.0, 0.0, 0.0, 0.0)
        
        cv2.putText(resized_frame, "TRACKER FAILED", 
                    (resized_frame.shape[1]//2, resized_frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
        
        print("TRACKER FAILED AT FRAME {}".format(frame_counter)) 
        cv2.imshow("Frame",resized_frame)
        cv2.waitKey(1)

        #Store aberrant post_location
        post_location.append(top_left[0])

    unfinished, frame = contraction_vid.read()
    frame_counter += 1


cv2.destroyAllWindows()

#From post locations determine if post is contracting to 'right' or 'left' 
#i.e. in direction of increasing or decreasing pixel location.'''

rightmost = max(post_location)
leftmost = min(post_location)
normalized_post_location = []

for i in post_location:
    
    normalized_post_location.append(abs(post_location[0] - i))
    
#Assuming video starts with relaxed tissue, post will be farthest from initial
#location during a contraction. If post contracts to right,
#the difference between rightmost and the initial post location will be much
#greater than the difference between leftmost and the intial post location. 
#And vice-versa if post contracts to the left.
   
if rightmost - post_location[0] > post_location[0] - leftmost:
    contracts_to_right = True
elif post_location[0] - leftmost > rightmost - post_location[0]:
    contracts_to_right = False
else:
    print("Unable to determine direction of post movement")
    contracts_to_right = True
        
multiple_contractions = input("Multiple contraction video? [Y/N] ")
max_displacement = -1
displacements = []

#If tissue contracts to right, maxima = contractions, minima = relaxations. 
#If tissue contracts to left, maxima = relaxations and minima = contractions.
maxima_indices = []
minima_indices = []

#Error factor for identifying local maxima and minima. Adjust if necessary 
error = 2

if multiple_contractions.upper() == 'N': 
    
    max_displacement = abs(max(post_location) - min(post_location))
    print("Contraction displacement = {} pixels.".format(max_displacement))  
    
    
else:
    
    mins = [post_location[0]]
    minima_indices.append(0)
    maxes = []
    
    contracting = True
    
    if contracts_to_right:
        
        #Initialize running max/min to values which will pass first test
        running_max = -1
        running_min = 1E6
        
        for i in range(len(post_location)):
            
            if i + 3 >= len(post_location):
                break
            
            #Store next three post locations
            next1, next2, next3 = (post_location[i + 1], 
            post_location[i + 2], post_location[i + 3])
            
            if contracting:
               
                if post_location[i] >= running_max:
                    running_max = post_location[i]
                    running_max_index = i
                
                #If tissue has begun relaxing/stopped contracting, store
                #running_max as true local maximum. Error margin included in
                #case of bbox drift.
            
                if (max([next1, next2, next3, running_max]) == running_max and 
                running_max - min([next1, next2, next3])) >= error:
                    contracting = False
                    maxes.append(post_location[running_max_index])
                    maxima_indices.append(running_max_index)
                    running_min = 1E6
                    
            elif not contracting:
                
                if post_location[i] <= running_min:
                    running_min = post_location[i]
                    running_min_index = i   
                
                
                #If tissue has begun contracting/stopped relaxing, store
                #running_min as true local maximum. Error margin included in
                #case of bbox drift.
                
                if (min([next1, next2, next3, running_min]) == running_min and
                max([next1, next2, next3]) - running_min) >= error:
                    contracting = True
                    mins.append(post_location[running_min_index])
                    minima_indices.append(running_min_index)
                    running_max = -1
                    
                    
    #If contracts to left, post location will decrease with contraction
    elif not contracts_to_right:
        
        
        #Initialize running max/min to values which will pass first test
        running_max = 1E6
        running_min = -1
        
        for i in range(len(post_location)):
            
            if i + 3 >= len(post_location):
                break
            
            #Store next three post locations
            next1, next2, next3 = (post_location[i + 1], 
            post_location[i + 2], post_location[i + 3])
            
            if contracting:
               
                if post_location[i] <= running_max:
                    running_max = post_location[i]
                    running_max_index = i
                
                #If tissue has begun relaxing/stopped contracting, store
                #running_max as true local maximum. Error margin included in
                #case of bbox drift.
            
                if (min([next1, next2, next3, running_max]) == running_max and 
                max([next1, next2, next3]) - running_max >= error):
                    contracting = False
                    maxes.append(post_location[running_max_index])
                    maxima_indices.append(running_max_index)
                    running_min = -1
                    
            elif not contracting:
                
                if post_location[i] >= running_min:
                    running_min = post_location[i]
                    running_min_index = i   
                
                
                #If tissue has begun contracting/stopped relaxing, store
                #running_min as true local maximum. Error margin included in
                #case of bbox drift.
                
                if (max([next1, next2, next3, running_min]) == running_min and
                running_min - min([next1, next2, next3])) >= error:
                    contracting = True
                    mins.append(post_location[running_min_index])
                    minima_indices.append(running_min_index)
                    running_max = 1E6
        

    for i in range(len(maxes)):
        
        #Default: displacement = contraction - most recent relaxation. 
        # This was used for all hMMT experiments. 
        
        displacements.append(abs(maxes[i] - mins[i]))
        
        #Alternative: displacement = contraction - initial relaxed state
        #displacements.append(abs(maxes[i] - mins[0]))
        
    output = ('''Maxes: {}, Mins: {}\nRelative displacements: {}\nContracted to right: {}'''
              .format(maxes, mins, displacements, contracts_to_right))
    
    extra_output = ""
    
    if multiple_contractions.upper() == 'BOTH':
        max_displacement = abs(max(post_location) - min(post_location))
        
        extra_output = '''\nIf single video method, contraction displacement = 
        {} pixels.'''.format(max_displacement)
    
    output = output + extra_output
    
    out_file = open("postTracking.txt", "w")
    out_file.write(output)
    out_file.close()
    
    print(output)

plt.plot(normalized_post_location)
plt.show()
plt.pause(0.001)
#If running from Command Prompt/Terminal, close all GUI windows to progress

export = input("Export post locations as .csv file? [Y/N] ")

if export.upper() == "Y":
    csv_file = open("postTracking.csv", "w")
    csv_locations = ""
    csv_locations += "Normalized post locations, Raw post locations,\n"
    for i in enumerate(normalized_post_location):
        csv_locations += str(i[1]) + "," + str(post_location[i[0]]) + ",\n"
        
    csv_file.write(csv_locations)
    csv_file.close()
