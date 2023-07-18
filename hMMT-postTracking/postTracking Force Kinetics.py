#!/usr/bin/env python3

"""
RateCounter2
Import post positions(csv) to calculate:
    time of contraction
    time to half-relaxation
@author : Zhuoye (Yvonne) Xie
date: 2022.10.1
"""

from tkinter import filedialog
import pandas as pd
import numpy as np

# Part 1: Import post locations

# CSV file selected by user manually for each run
f_path = filedialog.askopenfilename(initialdir="D:\"",
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
# For half-relaxation
mid_relax = []
mid_indices = []
half_time = []

error = 2

# Assume all tissues start at rest in the videos
if multiple_contractions.upper() == 'N' :

    # Make sure the maximum is among the whole video
    max_displacement = abs(max(post_loc) - min(post_loc))

    # If contracting to the right, find the last minimum and the first maximum
    # last minimum: the following 2 locations are larger
    # first maximum: the previous 2 locations are smaller 

    contracting = True
    resting = True

    if contracts_to_right:

        # Initialize two values to pass first test
        running_max = -1
        running_min = 1E6

        # Find the index of maximum post location
        for m in range(len(post_loc)):

            # Though this situation should never happen
            if m + 3 >= len(post_loc):
                break
            
            next1, next2, next3 = (post_loc[m + 1],
            post_loc[m + 2], post_loc[m+3])
            
            if contracting:

                if post_loc[m] >= running_max:
                    running_max = post_loc[m]
                    running_max_index = m

                if (max([next1, next2, next3, running_max])) == running_max and running_max - min([next1, next2, next3]) >= error:
                    contracting = False
                    rate_max = post_loc[running_max_index] # The max position
            
        # Find the index of mininum post location between start and max points
        start = post_loc[0]

        for n in range(1,running_max_index):

            if n + 3 >= running_max_index:
                break

            cur = post_loc[n]
            next1, next2, next3 = (post_loc[n + 1], post_loc[n + 2],post_loc[n + 3])

            if resting:

                if post_loc[n] <= running_min:
                    running_min = post_loc[n]
                    running_min_index = n
            
                if (min([start, cur, next1, next2, next3]) == running_min) and (max([next1, next2, next3]) - cur >= error):
                    resting = False
                    rate_min = post_loc[running_min_index] # The min position
        
        tetanus_time = abs(running_max_index - running_min_index) * (1/fps)         

    elif not contracts_to_right:

        # Initialize two values to pass first test
        running_max = -1
        running_min = 1E6

        # Find the index of maximum post location
        for m in range(len(post_loc)):

            # Though this situation should never happen
            if m + 3 >= len(post_loc):
                break
            
            next1, next2, next3 = (post_loc[m + 1],
            post_loc[m + 2], post_loc[m+3])
            
            if contracting:

                if post_loc[m] <= running_max:
                    running_max = post_loc[m]
                    running_max_index = m

                if (min([next1, next2, next3, running_max])) == running_max and max([next1, next2, next3]) - running_max >= error:
                    contracting = False
                    rate_max = post_loc[running_max_index] # The max position
            
        # Find the index of mininum post location between start and max points
        start = post_loc[0]

        for n in range(1,running_max_index):

            if n + 3 >= running_max_index:
                break

            cur = post_loc[n]
            next1, next2, next3 = (post_loc[n + 1], post_loc[n + 2],post_loc[n + 3])

            if resting:

                if post_loc[n] >= running_min:
                    running_min = post_loc[n]
                    running_min_index = n
            
                if (max([start, cur, next1, next2, next3]) == running_min) and cur - (min([next1, next2, next3]) >= error):
                    resting = False
                    rate_min = post_loc[running_min_index] # The min position
        
        tetanus_time = abs(running_max_index - running_min_index) * (1/fps)         

    print("Contraction displacement = {} pixels.".format(max_displacement))
    print("Contraction time = {}s. ".format(tetanus_time))

# For multiple contraction
else:
    
    cont_time = []

    maxima_indices = []
    minima_indices = []

    mins = [post_loc[0]]
    minima_indices.append(0)
    maxes = []
    
    contracting = True
    resting = True
    
    if contracts_to_right:
        
        #Initialize running max/min to values which will pass first test
        running_max = -1
        running_min = 1E6
        
        for i in range(len(post_loc)):
            
            if i + 3 >= len(post_loc):
                break
            
            next1, next2, next3 = (post_loc[i + 1], 
            post_loc[i + 2], post_loc[i + 3])
            
            if contracting:
               
                if post_loc[i] >= running_max:
                    running_max = post_loc[i]
                    running_max_index = i
                
                #If tissue has begun relaxing/stopped contracting, store
                #running_max as true local maximum. Error margin included in
                #case of bbox drift.
            
                if (max([next1, next2, next3, running_max]) == running_max and 
                running_max - min([next1, next2, next3])) >= error:
                    contracting = False
                    maxes.append(post_loc[running_max_index])
                    maxima_indices.append(running_max_index)
                    running_min = 1E6
                    
            elif not contracting:
                
                if post_loc[i] <= running_min:
                    running_min = post_loc[i]
                    running_min_index = i   
                
                
                #If tissue has begun contracting/stopped relaxing, store
                #running_min as true local maximum. Error margin included in
                #case of bbox drift.
                
                if (min([next1, next2, next3, running_min]) == running_min and
                max([next1, next2, next3]) - running_min) >= error:
                    contracting = True
                    mins.append(post_loc[running_min_index])
                    minima_indices.append(running_min_index)
                    running_max = -1
        
        # The first contraction time should not be relative the start time but the actual start point of contraction
        start = post_loc[0]
        running_min = start

        for n in range(1,maxima_indices[0]):

            if n + 3 >= maxima_indices[0]:
                break

            cur = post_loc[n]
            next1, next2, next3 = (post_loc[n + 1], post_loc[n + 2],post_loc[n + 3])

            if resting:

                if post_loc[n] <= running_min:
                    running_min = post_loc[n]
                    minima_indices[0] = n
            
                if (min([start, cur, next1, next2, next3]) == running_min) and (max([next1, next2, next3]) - cur >= error):
                    resting = False
                    rate_min = post_loc[minima_indices[0]] # The min position                        
                    
    #If contracts to left, post location will decrease with contraction
    elif not contracts_to_right:
        
        running_max = 1E6
        running_min = -1
        
        for i in range(len(post_loc)):
            
            if i + 3 >= len(post_loc):
                break
            
            next1, next2, next3 = (post_loc[i + 1], 
            post_loc[i + 2], post_loc[i + 3])
            
            if contracting:
               
                if post_loc[i] <= running_max:
                    running_max = post_loc[i]
                    running_max_index = i
                
                #If tissue has begun relaxing/stopped contracting, store
                #running_max as true local maximum. Error margin included in
                #case of bbox drift.
            
                if (min([next1, next2, next3, running_max]) == running_max and 
                max([next1, next2, next3]) - running_max >= error):
                    contracting = False
                    maxes.append(post_loc[running_max_index])
                    maxima_indices.append(running_max_index)
                    running_min = -1
                    
            elif not contracting:
                
                if post_loc[i] >= running_min:
                    running_min = post_loc[i]
                    running_min_index = i   
                
                
                #If tissue has begun contracting/stopped relaxing, store
                #running_min as true local maximum. Error margin included in
                #case of bbox drift.
                
                if (max([next1, next2, next3, running_min]) == running_min and
                running_min - min([next1, next2, next3])) >= error:
                    contracting = True
                    mins.append(post_loc[running_min_index])
                    minima_indices.append(running_min_index)
                    running_max = 1E6   

        start = post_loc[0]
        running_min = start

        for n in range(1,maxima_indices[0]):

            if n + 3 >= maxima_indices[0]:
                break

            cur = post_loc[n]
            next1, next2, next3 = (post_loc[n + 1], post_loc[n + 2],post_loc[n + 3])

            if resting:

                if post_loc[n] >= running_min:
                    running_min = post_loc[n]
                    minima_indices[0] = n
            
                if (max([start, cur, next1, next2, next3]) == running_min) and cur - (min([next1, next2, next3]) >= error):
                    resting = False
                    rate_min = post_loc[minima_indices[0]] # The min position

    # Find all the indices of half-relaxation point
    for k in range(len(maxes)-1):
        # Find the mid_point cloest to half-relaxation
        # It always use the index that first comes cloest to the point
        # Thus, the estimation is always a bit smaller than the actual value
        diff_min = 1E6

        if contracts_to_right:
            tmp = maxes[k] - (abs(maxes[k]-mins[k+1]))/2
            mid_relax.append(tmp)

            for mi in range(maxima_indices[k], minima_indices[k+1]):
                if (post_loc[mi] - tmp) < diff_min:
                    diff_min = post_loc[mi] - tmp
                    mid_index = mi

            mid_indices.append(mid_index)
        
        else:
            tmp = (abs(maxes[k])-mins[k+1])/2 + maxes[k]
            mid_relax.append(tmp)

            for mi in range(minima_indices[k], maxima_indices[k+1]):
                if (post_loc(mi) - tmp) < diff_min:
                    diff_min = post_loc[mi] - tmp
                    mid_index = mi
            
            mid_indices.append(mid_index)

    # Calculate displacements, full contraction time, and half-relaxation time
    for k in range(len(maxes)):
        displacements.append(abs(maxes[k] - mins[k]))
        cont_time.append((abs(maxima_indices[k] - minima_indices[k])) * (1/fps))
        if k < (len(maxes)-1):
            half_time.append(abs(maxima_indices[k] - mid_indices[k]) * (1/fps))

    # Round contraction time to 2 decimal place
    r_conttime = list(np.around(cont_time, 2))
    r_halftime = list(np.around(half_time, 2))

    # Integrate all data as output
    output = ('''Maxes: {}, Mins: {}\nRelative displacements: {}\nContracted to right: {}\nContraction time: {}\nHalf relaxation time: {} '''.format(maxes, mins, displacements, contracts_to_right, r_conttime, r_halftime))

    print(output)

