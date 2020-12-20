

# get duration of video in milliseconds
def get_video_length(filename):
    from moviepy.editor import VideoFileClip
    
    clip = VideoFileClip(video_file)
    file_duration = int(clip.duration*1000)
    return(file_duration)