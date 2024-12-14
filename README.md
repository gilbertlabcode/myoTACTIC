# MyoTACTIC-postTracking

## Overview
This repository hosts Python scripts that track and analyze the deflections of MyoTACTIC microposts during human muscle microtissue (hMMT) contraction events.

## Scripts Description

### `postTracking.py`
- **Input**: Video file capturing micropost deflections.
- **Outputs**:
  - Raw post position data
  - Relative displacement

### `postTracking+Kinetics.py`
- **Input**: Video file capturing micropost deflections.
- **Outputs**:
  - Raw post position data
  - Relative displacement
  - Contraction kinetics metrics
  - Indices of the frames used for calculating contraction kinetics metrics

### `RateCounter.py`
- **Input**: Raw post position data (provided as `.csv` file).
- **Outputs**:
  - Relative displacement
  - Contraction kinetics metrics
  - Indices of the frames used for calculating contraction kinetics metrics

## Contraction Metrics
The following are the contraction metrics output by the scripts:
- **Relative Displacement (pixels)**: Measure of the difference in post position from the peak to nadir of contraction
- **Time-to-Peak Twitch (seconds)**: Time elapsed from the onset of contraction to its peak
- **Duration-at-Peak (seconds)**: Duration for which the contraction remains at its peak value
- **Half Relaxation Time (seconds)**: Time taken for the contraction to decrease to half of its peak value
- **Contraction Rate (pixels/second)**: Speed of the contraction phase
- **Relaxation Rate (pixels/second)**: Speed of the relaxation phase
- **Full Width at Half Max (seconds)**: Duration of the contraction at half its maximum amplitude
