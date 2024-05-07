# -*- coding: utf-8 -*-
"""
postTracker2024
To obtain post positions and contraction parameters of a muscle tissue from a contraction video.
Outputs include the contraction parameters, indices of points used for parameter calculation, and post position data.
List of parameters:
    Relative Displacement (DOF)(pxl)
    Time-to-Peak Twitch (TPT) (sec)
    Duration-at-Peak (DP) (sec)
    Half Relaxation Time (1/2 RT)(sec)
    Contraction Rate (pxl/sec)
    Relaxation Rate (pxl/sec)
    Full Width at Half Max (sec)
List of indices:
    Start of Contraction (cont_start) - P1
    Mid-contraction (cont_mid) - P2
    Beginning point of peak (peak_start) - P3
    End point of peak (peak_end) - P4
    Mid-relaxation (relax_mid) - P5
    Termination of Contraction (cont_term) - P6

@author: Zhuoye (Yvonne) Xie
Updated from postTracking.py by Haben Y. Abraha
Last Updated Date: 2023-08-28
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import cv2
import sys
import matplotlib
matplotlib.use("TkAgg")


class ClickLocation:

    def __init__(self):
        self.roi_center = None

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_center = (x, y)


vid_path = filedialog.askopenfilename(
    initialdir="D:\"",
    filetypes=(("All Files", "*.*"), ("Text File", "*.txt")),
    title="Choose a file")

contraction_vid = cv2.VideoCapture(vid_path)

# Define frame per second of the video.
fps = 30

colour = (255, 50, 50)
frame_counter = 1
user_roi = False
gray_history = []
fx = 0.55
fy = 0.55

# Extract the first fram of the video for ROI selection.
ok, first_frame = contraction_vid.read()
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.equalizeHist(first_frame_gray)

if not ok:
    print("Cannot read the first frame")
    sys.exit()

'''
Due to screen size constraints, images shown to user are all resized versions 
of the frames that the tracking is done on. 
ROI format is:
(top_left_corner_x_coord, top_left_corner_y_coord, ROI_width, ROI_height).

'''

# ROI width/height lengths. Choose a pair or adjust the values if necessary.

# Larger ROI - less accurate (more susceptible to noise) but requires lower quality videos.
# Smaller ROI - more accurate but requires higher quality videos.

#RHF = 0.25
#RWF = 0.08

RHF = 0.1
RWF = 0.03

#RHF = 0.2
#RWF = 0.06

resized_frame = cv2.resize(first_frame, (0, 0), fx=fx, fy=fy)

cv2.namedWindow("Identify post")
clickLocation = ClickLocation()
cv2.setMouseCallback("Identify post", clickLocation.click)

while (True):

    marked_frame = resized_frame.copy()
    if not clickLocation.roi_center is None:
        cv2.circle(marked_frame, (clickLocation.roi_center[0],
                        clickLocation.roi_center[1]), 5, (255, 0, 0), 1)

        # Get user click location to draw ROI on screen.
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

print("ROI size: {},{}\nUser ROI: {}\nROI corner position: {}".format(
    RHF, RWF, user_roi, roi))

# Draw a rectangle for visualization of the selected ROI.
cv2.rectangle(resized_frame, (user_top_left), (user_bottom_right), colour, 1)
cv2.imshow("Frame", resized_frame)
cv2.moveWindow("Frame", 0, 0)

cv2.waitKey(0)
cv2.destroyWindow("Frame")

tracker = cv2.TrackerCSRT_create()

# Initialize the tracker with first frame.
tracker.init(first_frame_gray, roi)
gray_history.append(first_frame_gray)

# Record the first x-axis location of post, i.e. center of roi
post_location = [roi[0] + roi[2]/2]

# Go to the second frame
unfinished, frame = contraction_vid.read()

while unfinished:

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    resized_frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)
    tracked, roi = tracker.update(frame_gray)
    gray_history.append(frame_gray)

    if tracked:
        # Tracker has identified post.

        # Roi points are floating point (cast to int for use).

        top_left = (int(roi[0]), int(roi[1]))
        bottom_right = (int(roi[0]) + int(roi[2]), int(roi[1]) + int(roi[3]))

        resized_top_left = (round(top_left[0]*fx), round(top_left[1]*fy))
        resized_bottom_right = (resized_top_left[0] + round(roi[2]*fx),
                                resized_top_left[1] + round(roi[3]*fy))

        # Resize roi for depicting with 'imshow' to fit screen.
        cv2.rectangle(resized_frame, resized_top_left, resized_bottom_right,
                      colour, 1)
        cv2.imshow("Frame", resized_frame)
        cv2.waitKey(1)

        # Record x-axis location of post.
        post_location.append(top_left[0] + roi[2]/2)
        
    else:

        # If tracking failed, roi becomes = (0.0, 0.0, 0.0, 0.0).

        cv2.putText(resized_frame, "TRACKER FAILED",
                    (resized_frame.shape[1]//2, resized_frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        print("TRACKER FAILED AT FRAME {}".format(frame_counter))
        cv2.imshow("Frame", resized_frame)
        cv2.waitKey(1)

        # Store aberrant post_location.
        post_location.append(top_left[0])

    unfinished, frame = contraction_vid.read()
    frame_counter += 1


cv2.destroyAllWindows()

# From post locations determine if post is contracting to 'right' or 'left'.
# i.e. in direction of increasing or decreasing pixel location.
rightmost = max(post_location)
leftmost = min(post_location)
normalized_post_location = []

for i in post_location:

    # Note that, the normalized positions take the absolute difference between
    # current post position and the initial post position, which means if the
    # post goes over the initial position, the normalized position may not show
    # this properly.
    normalized_post_location.append(abs(post_location[0] - i))

# Assuming video starts with relaxed tissue, post will be farthest from 
# initial location during a contraction. If post contracts to right,
# the difference between rightmost and the initial post location will be much
# greater than the difference between leftmost and the intial post location.
# And vice-versa if post contracts to the left.

if rightmost - post_location[0] > post_location[0] - leftmost:
    contracts_to_right = True
elif post_location[0] - leftmost > rightmost - post_location[0]:
    contracts_to_right = False
else:
    print("Unable to determine direction of post movement")
    contracts_to_right = True

# Ask the user whether the video contains multiple contractions.
multiple_contractions = input("Multiple contraction video? [Y/N] ")
max_displacement = -1
displacements = []

# Store the lists of indices for each point of interest during contractions.
peak_end = []
cont_start = []
peak_start = []
cont_mid = []
relax_mid = []

# Error factor for identifying local maxima and minima. Adjust if necessary.
# Error is used including the equal situation.
error = 2

if multiple_contractions.upper() == 'N':

    # Make sure the maximum is among the whole video.
    max_displacement = abs(max(post_location) - min(post_location))

    # Assume the tissue in the video starts at rest
    contracting = False

    if contracts_to_right:

        running_max = -1
        
        # Find the index of maximum post location
        for m in range(1, len(post_location)):

            # Though this situation should never happen
            if m + 3 >= len(post_location):
                break

            cur = post_location[m]
            next1, next2, next3 = (post_location[m + 1],
                                   post_location[m + 2], post_location[m + 3])

            if contracting:
                
                if cur > running_max:
                    running_max = cur
                    running_max_index = m

            else:

                if (min([next1, next2, next3]) > cur and (max([next2, next3]) - 
                                                          cur >= error)):
                    contracting = True
                    rate_min = cur

    elif not contracts_to_right:
        
        running_max = 1E6
        
        # Find the index of maximum post location
        for m in range(1, len(post_location)):

            # Though this situation should never happen
            if m + 3 >= len(post_location):
                break

            cur = post_location[m]
            next1, next2, next3 = (post_location[m + 1],
                                   post_location[m + 2], post_location[m + 3])

            if contracting:

                if cur < running_max:
                    running_max = cur
                    runnning_max_index = m

            else:

                if (min([next1, next2, next3]) < cur and (max([next2, next3]) -        
                                                           cur >= error)):
                    contracting = True
                    rate_min = cur

    tetanus_time = abs(running_max_index - rate_min) * (1/fps)

    print("Contraction displacement = {} pixels.".format(max_displacement))
    print("Contraction time = {} s. ".format(tetanus_time))
    print("Rate of contraction = {} pixel/s".format(max_displacement /
                                                    tetanus_time))

else:
    
    # mins are start of contraction position and maxes are peak position
    mins = []
    maxes = []

    # The video should start at rest.
    contracting = False

    if contracts_to_right:

        for i in range(1, len(post_location)):

            if i + 3 >= len(post_location):
                break

            # Store next three post locations
            cur = post_location[i]
            next1, next2, next3 = (post_location[i + 1],
                                   post_location[i + 2], post_location[i + 3])

            if contracting:

                # If tissue has begun relaxing/stopped contracting, store
                # running_max as true local maximum. Error margin included in
                # case of bbox drift.
                
                # The end of contraction has to be larger than all the next 3 
                # positions.
                if (cur > max([next1, next2, next3])):
                    contracting = False
                    maxes.append(cur)
                    peak_end.append(i)

            elif not contracting:

                # If tissue has begun contracting/stopped relaxing, store
                # running_min as true local minimum. Error margin included in
                # case of bbox drift.

                if (min([next1, next2, next3]) > cur and
                    max([next1, next2, next3]) - cur) >= error:
                    contracting = True
                    mins.append(cur)
                    cont_start.append(i)

    # If contracts to left, post location will decrease with contraction
    elif not contracts_to_right:

        for i in range(len(post_location)):

            if i + 3 >= len(post_location):
                break

            # Store next three post locations
            cur = post_location[i]
            next1, next2, next3 = (post_location[i + 1],
                                   post_location[i + 2], post_location[i + 3])

            if contracting:

                # If tissue has reached the end of peak contraction, store
                # current index i as true local maximum. 

                if (min([next1, next2, next3]) > cur):
                    contracting = False
                    maxes.append(cur)
                    peak_end.append(i)

            elif not contracting:

                # If tissue has started to contract, store the current index
                # i as true local maximum. Error margin included in case of 
                # bbox drift.

                if max([next1, next2, next3]) < cur and (cur -  
                   min([next1, next2, next3]) >= error):
                    contracting = True
                    mins.append(cur)
                    cont_start.append(i)

    # Find all the indices of the start of the peak
    for k in range(len(maxes)):
        
        if contracts_to_right:
            
            running_max = -1
            
            for j in range(cont_start[k], peak_end[k] + 1):
                if post_location[j] > running_max:
                    running_max = post_location[j]
                    first_max = j
            
        else:
            
            running_max = 1E6
            
            for j in range(cont_start[k], peak_end[k] + 1):
                if post_location[j] < running_max:
                    running_max = post_location[j]
                    first_max = j
                    
        peak_start.append(first_max)

    # Find all the end of relaxation points
    cont_term = []
    for k in range(len(mins) - 1):
        
        diff_min = 1E6
        
        if contracts_to_right:
    
            for j in range(peak_end[k], cont_start[k + 1] + 1):
                
                diff = post_location[j] - mins[k + 1]
                if diff < diff_min and diff >= 0:
                    diff_min = diff
                    relax_index = j
            
        else:
            
            for j in range(peak_end[k], cont_start[k + 1] + 1):
                
                diff = post_location[j] - mins[k + 1]
                if abs(diff) < diff_min and diff <= 0: 
                    diff_min = abs(diff)
                    relax_index = j
                
        cont_term.append(relax_index)   
    
    # Find all the indices of half-contraction and half-relaxation points
    half_cont_index = 0
    for m in range(0, len(cont_start) - 1):
        # Find the points cloest to half-contraction and used the same post position to find half_relaxation points
        # It always use the index that first comes cloest to the point
        # Thus, the estimation would be no larger than the actual value
        
        if contracts_to_right:
            
            # Half point of contraction
            tmp1 = maxes[m] - (abs(maxes[m] - mins[m])) / 2
            tmp2 = maxes[m] - (abs(maxes[m] - post_location[cont_term[m]])) / 2

            # Half_contraction points
            diff_min = 1E6
            for mi in range(peak_start[m], cont_start[m], -1):
                if (abs(post_location[mi] - tmp1) < diff_min and 
                    post_location[mi] >= tmp1):
                    diff_min = abs(post_location[mi] - tmp1)
                    half_cont_index = mi
            
            # Corresponding half_relaxation points
            diff_min = 1E6
            for ni in range(peak_end[m], cont_start[m + 1]):
                if (abs(post_location[ni] - tmp2) < diff_min and 
                    post_location[ni] <= tmp2):
                    diff_min = abs(post_location[ni] - tmp2)
                    half_relax_index = ni

            cont_mid.append(half_cont_index)
            relax_mid.append(half_relax_index)
        
        else:
            
            tmp1 = (abs(mins[m]) - maxes[m]) / 2 + maxes[m]
            tmp2 = (abs(post_location[cont_term[m]]) - maxes[m]) / 2 + maxes[m]
            
            # Half_contraction points
            diff_min = 1E6
            for mi in range(peak_end[m], cont_start[m], -1):
                if (abs(post_location[mi] - tmp1) < diff_min and 
                    post_location[mi] <= tmp1):
                    diff_min = abs(post_location[mi] - tmp1)
                    half_cont_index = mi
            
            # Corresponding half_relaxation points
            diff_min = 1E6
            for ni in range(peak_end[m], cont_start[m + 1]):
                if (abs(post_location[ni] - tmp2) < diff_min and 
                    post_location[ni] >= tmp2):
                    diff_min = abs(post_location[ni] - tmp2)
                    half_relax_index = ni
                    
            cont_mid.append(half_cont_index)
            relax_mid.append(half_relax_index)
          
    # Calculate displacements, full contraction time, and half-relaxation time
    cont_time =[]
    half_width = []
    half_relax_time = []
    peak_duration = []
    
    for k in range(len(maxes)):
        
        # displacement = contraction - most recent relaxation. 
        # This was used for all hMMT experiments. 
        
        displacements.append(abs(maxes[k] - mins[k]))
        
        # Time to peak
        cont_time.append((abs(peak_start[k] - cont_start[k])) * (1/fps))
        
        # Peak duration
        peak_duration.append((peak_end[k] - peak_start[k]) * (1/fps))
        
        # Full width of half max
        if k < (len(cont_term) - 1):
            half_width.append(abs(cont_mid[k] - 
                                  relax_mid[k]) * (1/fps))
        
        if k < len(relax_mid):
            half_relax_time.append(abs(peak_end[k] - 
                                       relax_mid[k]) * (1/fps))
        
    rate_cont = []
    rate_relax = []
    for k in range(len(displacements)):
        rate_cont.append(displacements[k]/cont_time[k])
        
        if k < (len(cont_term)):
            disp_relax = abs(post_location[cont_term[k]] - 
                             post_location[peak_end[k]])
            time_relax = abs(cont_term[k] - peak_end[k]) * (1/fps)
            rate_relax.append(disp_relax/time_relax)
    
    # Round contraction time to 2 decimal place
    cont_time = list(np.around(cont_time, 2))
    peak_duration = list(np.around(peak_duration, 2))
    half_width = list(np.around(half_width, 2))
    rate_cont = list(np.around(rate_cont, 2))
    rate_relax = list(np.around(rate_relax, 2))
    half_relax_time = list(np.around(half_relax_time, 2))
    
    output = ('''Maxes: {}\nMins: {}\nRelative Displacement (DOF)(pxl): {}\nContracted to right: {}\nTime-to-Peak Twitch (TPT)(sec): {}\nDuration-at-Peak (DP)(sec): {}\nHalf relaxation time (1/2 RT)(sec): {}\nContraction Rate (pxl/sec): {}\nRelaxation Rate (pxl/sec): {}\nFull Width at Half Max (sec): {}\n'''
              .format(maxes, mins, displacements, contracts_to_right, cont_time, peak_duration, half_relax_time, rate_cont, rate_relax, half_width))
    
    indices_output = ('''P1: {}\nP2: {}\nP3: {}\nP4: {}\nP5: {}\nP6: {}\n'''
                      .format(cont_start, cont_mid, peak_start, peak_end, relax_mid, cont_term))
    
    extra_output = ""
    
    if multiple_contractions.upper() == 'BOTH':
        max_displacement = abs(max(post_location) - min(post_location))
        
        extra_output = '''\nIf single video method, contraction displacement = 
        {} pixels.'''.format(max_displacement)
    
    output = output + indices_output + extra_output
    
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
    

