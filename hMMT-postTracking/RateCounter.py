"""
RateCounter23
Import post positions(csv) to calculate:
    time of contraction
    time to half-relaxation
@author : Zhuoye (Yvonne) Xie
Last Updated Date: 2023-08-28
"""

from tkinter import filedialog
import pandas as pd
import numpy as np

# Part 1: Import post locations

# CSV file selected by user manually for each run
f_path = filedialog.askopenfilename(initialdir="C:\"",
        filetypes=(("All Files","*.*"), ("CSV Files","*.csv")), 
        title= "Choose a CSV file.")

# Select one colomn as post locations 
n = int(input("Which column to use? Input an integer. "))

# Import the column and name the column as 'loc'
# "skiprows = 1" here means the column of data selected has its column name in the first row
# If the column has no column name, then delete "skiprows = 1" to use the data from the first row of the column
df = pd.read_csv(f_path, skiprows = 1, usecols = [n - 1], names = ['loc'])

# If don't want to select a file everytime, just put the file path as below
# df = pd.read_csv(r'C:\Users\admin\Desktop\akaka.csv', skiprows = 1, usecols = [n - 1], names = ['loc'])
# If want to input column before running the code, just change "n-1" in usecols to a number
# If want to use the second column, put "1", eg. usecols = [1] (since python start at 0)
# If want to use the theird column, put "2"

post_loc = df['loc'].tolist()

# Part2 : Measure contraction rate

# fps of the video
# User input
# fps = input("What is the fps of the video? Input an integer.")
# Set a default value
fps = 30

# From post location determine if the post is contracting to 'left' or 'right'
rightmost = max(post_loc)
leftmost = min(post_loc)

if (rightmost - post_loc[0]) > (post_loc[0] - leftmost):
    contracts_to_right = True
elif (post_loc[0] - leftmost) > (rightmost - post_loc[0]):
    contracts_to_right = False
else:
    print("Unable to determine contraction direction")
    contracts_to_right = True

# Measuring displacement and contraction time 
multiple_contractions = input("Multiple contraction video? [Y/N] ")
max_displacement = -1
displacements = []

# If tissue contracts to right, maxima = end of contractions, minima = end of relaxations.
# If tissue contracts to left, maxima = end of relaxations and minima = end of contractions.
maxima_indices = []
minima_indices = []
peak_indices = []
half_contract_indices = []
half_relax_indices = []

# Error factor for identifying local maxima and minima. Adjust if necessary.
# Error is used including the equal situation.
error = 2

if multiple_contractions.upper() == 'N':

    # Make sure the maximum is among the whole video
    max_displacement = abs(max(post_loc) - min(post_loc))

    # If contracting to the right, find the last minimum and the first maximum
    # last minimum: the following 2 locations are larger
    # first maximum: the previous 2 locations are smaller

    # Assume the tissue in the video starts at rest
    contracting = False

    if contracts_to_right:

        running_max = -1
        
        # Find the index of maximum post location
        for m in range(1, len(post_loc)):

            # Though this situation should never happen
            if m + 3 >= len(post_loc):
                break

            cur = post_loc[m]
            next1, next2, next3 = (post_loc[m + 1],
                                   post_loc[m + 2], post_loc[m + 3])

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
        for m in range(1, len(post_loc)):

            # Though this situation should never happen
            if m + 3 >= len(post_loc):
                break

            cur = post_loc[m]
            next1, next2, next3 = (post_loc[m + 1],
                                   post_loc[m + 2], post_loc[m + 3])

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

    mins = []
    maxes = []

    # The video should start at rest.
    contracting = False

    if contracts_to_right:

        for i in range(1, len(post_loc)):

            if i + 3 >= len(post_loc):
                break

            # Store next three post locations
            cur = post_loc[i]
            next1, next2, next3 = (post_loc[i + 1],
                                   post_loc[i + 2], post_loc[i + 3])

            if contracting:

                # If tissue has begun relaxing/stopped contracting, store
                # running_max as true local maximum. Error margin included in
                # case of bbox drift.
                
                # The end of contraction has to be larger than all the next 3 
                # positions.
                if (cur > max([next1, next2, next3])):
                    contracting = False
                    maxes.append(cur)
                    maxima_indices.append(i)

            elif not contracting:

                # If tissue has begun contracting/stopped relaxing, store
                # running_min as true local minimum. Error margin included in
                # case of bbox drift.

                if (min([next1, next2, next3]) > cur and
                    max([next1, next2, next3]) - cur) >= error:
                    contracting = True
                    mins.append(cur)
                    minima_indices.append(i)

    # If contracts to left, post location will decrease with contraction
    elif not contracts_to_right:

        for i in range(len(post_loc)):

            if i + 3 >= len(post_loc):
                break

            # Store next three post locations
            cur = post_loc[i]
            next1, next2, next3 = (post_loc[i + 1],
                                   post_loc[i + 2], post_loc[i + 3])

            if contracting:

                # If tissue has reached the end of peak contraction, store
                # current index i as true local maximum. 

                if (min([next1, next2, next3]) > cur):
                    contracting = False
                    maxes.append(cur)
                    maxima_indices.append(i)

            elif not contracting:

                # If tissue has started to contract, store the current index
                # i as true local maximum. Error margin included in case of 
                # bbox drift.

                if max([next1, next2, next3]) < cur and (cur -  
                   min([next1, next2, next3]) >= error):
                    contracting = True
                    mins.append(cur)
                    minima_indices.append(i)

    # Find all the indices of the start of the peak
    for k in range(len(maxes)):
        
        if contracts_to_right:
            
            running_max = -1
            
            for j in range(minima_indices[k], maxima_indices[k] + 1):
                if post_loc[j] > running_max:
                    running_max = post_loc[j]
                    first_max = j
            
        else:
            
            running_max = 1E6
            
            for j in range(minima_indices[k], maxima_indices[k] + 1):
                if post_loc[j] < running_max:
                    running_max = post_loc[j]
                    first_max = j
                    
        peak_indices.append(first_max)

    # Find all the end of relaxation points
    relax_indices = []
    for k in range(len(mins) - 1):
        
        diff_min = 1E6
        
        if contracts_to_right:
    
            for j in range(maxima_indices[k], minima_indices[k + 1] + 1):
                
                diff = post_loc[j] - mins[k + 1]
                if diff < diff_min and diff >= 0:
                    diff_min = diff
                    relax_index = j
            
        else:
            
            for j in range(maxima_indices[k], minima_indices[k + 1] + 1):
                
                diff = post_loc[j] - mins[k + 1]
                if abs(diff) < diff_min and diff <= 0: 
                    diff_min = abs(diff)
                    relax_index = j
                
        relax_indices.append(relax_index)   
    
    # Find all the indices of half-contraction and half-relaxation points
    half_cont_index = 0
    for m in range(0, len(relax_indices) - 1):
        # Find the points cloest to half-contraction and used the same post position to find half_relaxation points
        # It always use the index that first comes cloest to the point
        # Thus, the estimation would be no larger than the actual value
        
        if contracts_to_right:
            
            # Half point of contraction
            tmp1 = maxes[m] - (abs(maxes[m] - mins[m])) / 2
            tmp2 = maxes[m] - (abs(maxes[m] - post_loc[relax_indices[m]])) / 2
            
            # Half_contraction points
            diff_min = 1E6
            for mi in range(peak_indices[m], minima_indices[m], -1):
                if (abs(post_loc[mi] - tmp1) < diff_min and 
                    post_loc[mi] >= tmp1):
                    diff_min = abs(post_loc[mi] - tmp1)
                    half_cont_index = mi
            
            # Corresponding half_relaxation points
            diff_min = 1E6
            for ni in range(maxima_indices[m], minima_indices[m + 1]):
                if (abs(post_loc[ni] - tmp2) < diff_min and 
                    post_loc[ni] <= tmp2):
                    diff_min = abs(post_loc[ni] - tmp2)
                    half_relax_index = ni

            half_contract_indices.append(half_cont_index)
            half_relax_indices.append(half_relax_index)
        
        else:
            
            tmp1 = (abs(mins[m]) - maxes[m]) / 2 + maxes[m]
            tmp2 = (abs(post_loc[relax_indices[m]]) - maxes[m]) / 2 + maxes[m]
            
            # Half_contraction points
            diff_min = 1E6
            for mi in range(maxima_indices[m], minima_indices[m], -1):
                if (abs(post_loc[mi] - tmp1) < diff_min and 
                    post_loc[mi] <= tmp1):
                    diff_min = abs(post_loc[mi] - tmp1)
                    half_cont_index = mi
            
            # Corresponding half_relaxation points
            diff_min = 1E6
            for ni in range(maxima_indices[m], minima_indices[m + 1]):
                if (abs(post_loc[ni] - tmp2) < diff_min and 
                    post_loc[ni] >= tmp2):
                    diff_min = abs(post_loc[ni] - tmp2)
                    half_relax_index = ni
                    
            half_contract_indices.append(half_cont_index)
            half_relax_indices.append(half_relax_index)
        
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
        cont_time.append((abs(peak_indices[k] - minima_indices[k])) * (1/fps))
        
        # Peak duration
        peak_duration.append((maxima_indices[k] - peak_indices[k]) * (1/fps))
        
        # Full width of half max
        if k < (len(relax_indices) - 1):
            half_width.append(abs(half_contract_indices[k] - 
                                  half_relax_indices[k]) * (1/fps))
        
        if k < len(half_relax_indices):
            half_relax_time.append(abs(maxima_indices[k] - 
                                       half_relax_indices[k]) * (1/fps))
        
    rate_cont = []
    rate_relax = []
    for k in range(len(displacements)):
        rate_cont.append(displacements[k]/cont_time[k])
        
        if k < (len(relax_indices)):
            disp_relax = abs(post_loc[relax_indices[k]] - 
                             post_loc[maxima_indices[k]])
            time_relax = abs(relax_indices[k] - maxima_indices[k]) * (1/fps)
            rate_relax.append(disp_relax/time_relax)
    
    # Round contraction time to 2 decimal place
    cont_time = list(np.around(cont_time, 2))
    peak_duration = list(np.around(peak_duration, 2))
    half_width = list(np.around(half_width, 2))
    rate_cont = list(np.around(rate_cont, 2))
    rate_relax = list(np.around(rate_relax, 2))
    half_relax_time = list(np.around(half_relax_time, 2))
    
    output = ('''Maxes: {}\nMins: {}\nRelative displacements: [{}]\nContracted to right: {}\nTime to Max Twitch: {}\nPeak duration: {}\nHalf relaxation time: {}\nRate of contraction: {}\nRate of relaxation: {}\nFull width of half max: {}'''
              .format(
                  maxes, 
                  mins, 
                  ', '.join([f'{x:.2f}' for x in displacements]),  # Join list elements into a string
                  contracts_to_right, 
                  cont_time, 
                  peak_duration, 
                  half_relax_time, 
                  rate_cont, 
                  rate_relax, 
                  half_width
              ))
    
    extra_output = ""
    
    if multiple_contractions.upper() == 'BOTH':
        max_displacement = abs(max(post_loc) - min(post_loc))
        
        extra_output = '''\nIf single video method, contraction displacement = 
        {} pixels.'''.format(max_displacement)
    
    output = output + extra_output
    
    out_file = open("postTracking.txt", "w")
    out_file.write(output)
    out_file.close()
    
    print(output)
    
    # Uncomment for manual indices check
    print(minima_indices)
    print(half_contract_indices)
    print(peak_indices)
    print(maxima_indices)
    print(half_relax_indices)
    print(relax_indices)
    
    