# The EmoCodes Library
* [About](http://github.com/catcamacho/emocodes#about)
* [Installation](http://github.com/catcamacho/emocodes#installation)
* [Examples](http://github.com/catcamacho/emocodes#example-uses)
* [Coming Soon](http://github.com/catcamacho/emocodes#coming-soon)

## About
The EmoCodes library was designed to be a companion tool to the [EmoCodes](https://osf.io/xte7u/) video coding system.  This library can be used to do a variety of tasks, however, from video processing to converting Datavyu codes for other analyses.

For the code module, the assumption is that the input is a [Datavyu](https://datavyu.org/) style CSV.

## Installation
Emocodes requires python>=3.6. We recommend you install using pip:

```pip install emocodes```

This should install emocodes and all its dependencies.
## Example Uses
### Validate Datavyu codes
This example takes a Datavyu CSV and check for the following common errors:
* Offsets after onsets
* Sections of video not coded
* Offsets of zero
* Codes that to do not reach the end of the video or extend beyond the end of the video


### Convert Datavyu codes to code time series
This example takes a Datavyu CSV ('mycodes.csv') and converts the codes to a timeseries for use in other analyses (such as neuroimaging).

```
import emocodes as ec

# create instance of a CodeSeries class and set sampling rate for the output to 5Hz
cts = ec.CodeTimeSeries(sampling_rate=5)  

# use proc_codes_file to convert your Datavyu codes to a time series
cts.proc_codes_file(codes_file='mycodes.csv', video_file='video.mp4')
```

### extract features from a video
This example shows haw to take a video file and extract:
 * a mean luminance time series
 * RGB time series

```
import emocodes as ec

# find duration of your video
duration = ec.get_video_length('video.mp4')

# create dataframe "lum" with luminance and RGB values
lum = ec.compute_luminance('video.mp4', sampling_rate=5, duration)

# save the dataframe as a csv
lum.to_csv('luminance.csv')
```

## Coming Soon
* Point-and-click graphic user interface to enhance accessibility
* Wiki with API and example use scripts
