from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: video_transcoding
====
Examples: 68784
====
URL: https://www.openml.org/search?type=data&id=44974
====
Description: **Data Description**

The dataset contains a million randomly sampled video instances listing 10 fundamental video characteristics along with the YouTube video ID.

The videos were all transcribed from one format into another, measuring the memory usage and the transcription time.

The goal is to predict the transcription time using the input information and the desired output format.

**Attribute Description**

1. *id* - Youtube video id (should be dropped for the analysis)
2. *duration* - duration of video
3. *codec* - coding standard used for the video ("mpeg4", "h264", "vp8", "flv")
4. *width* - width of video in pixles
5. *height* - height of video in pixles
6. *bitrate* - video bitrate
7. *framerate* - actual video frame rate
8. *i* - number of i frames in the video
9. *p* - number of p frames in the video
10. *b* - number of b frames in the video
11. *frames* - number of frames in video
12. *i_size* - total size in byte of i videos
13. *p_size* - total size in byte of p videos
14. *b_size* - total size in byte of b videos
15. *size* - total size of video
16. *o_codec* - output codec used for transcoding ("mpeg4", "h264", "vp8", "flv")
17. *o_bitrate* - output bitrate used for transcoding
18. *o_framerate* - output framerate used for transcoding
19. *o_width* - output width in pixel used for transcoding
20. *o_height* - output height used in pixel for transcoding
21. *umem* -  total codec allocated memory for transcoding, alternate target feature
22. *utime* - total transcoding time for transcoding, target feature
====
Target Variable: utime (numeric, 10960 distinct): ['1.244', '0.964', '1.24', '1.288', '1.26', '0.968', '0.988', '0.672', '1.232', '1.056']
====
Features:

duration (numeric, 1086 distinct): ['130.3567', '74.535', '273.816', '143.0333', '256.2083', '385.052', '292.092', '33.09', '391.08', '77.233']
codec (nominal, 4 distinct): ['h264', 'vp8', 'mpeg4', 'flv']
width (numeric, 6 distinct): ['480.0', '320.0', '176.0', '1280.0', '640.0', '1920.0']
height (numeric, 6 distinct): ['360.0', '240.0', '144.0', '720.0', '480.0', '1080.0']
bitrate (numeric, 1095 distinct): ['54590.0', '406908.0', '1571003.0', '51082.0', '57565.0', '441862.0', '480125.0', '279173.0', '230847.0', '382461.0']
framerate (numeric, 261 distinct): ['29.0', '12.0', '25.0', '30.0', '15.0', '23.0', '24.0', '7.0', '13.0', '16.0']
i (numeric, 306 distinct): ['37.0', '23.0', '51.0', '15.0', '53.0', '112.0', '110.0', '87.0', '113.0', '77.0']
p (numeric, 1042 distinct): ['6726.0', '4646.0', '7541.0', '3271.0', '6457.0', '434.0', '4826.0', '6766.0', '3018.0', '9155.0']
b (numeric, 20 distinct): ['0.0', '704.0', '626.0', '2416.0', '1539.0', '1996.0', '6.0', '891.0', '1251.0', '674.0']
frames (numeric, 1044 distinct): ['1050.0', '1716.0', '1318.0', '2862.0', '2318.0', '1184.0', '792.0', '3074.0', '9232.0', '8755.0']
i_size (numeric, 1099 distinct): ['64483.0', '1776729.0', '6190779.0', '56405.0', '302900.0', '16244376.0', '2375028.0', '101794.0', '9023743.0', '182297.0']
p_size (numeric, 1099 distinct): ['825054.0', '6254165.0', '47579961.0', '856907.0', '1540700.0', '5023129.0', '15155062.0', '1052937.0', '2261252.0', '3510033.0']
size (numeric, 1099 distinct): ['889537.0', '8030894.0', '53770740.0', '913312.0', '1843600.0', '21267505.0', '17530090.0', '1154731.0', '11284995.0', '3692330.0']
o_codec (nominal, 4 distinct): ['mpeg4', 'vp8', 'flv', 'h264']
o_bitrate (numeric, 7 distinct): ['56000.0', '109000.0', '5000000.0', '3000000.0', '539000.0', '242000.0', '820000.0']
o_framerate (numeric, 5 distinct): ['15.0', '12.0', '29.97', '25.0', '24.0']
o_width (numeric, 6 distinct): ['176.0', '320.0', '480.0', '640.0', '1920.0', '1280.0']
o_height (numeric, 6 distinct): ['144.0', '240.0', '360.0', '480.0', '1080.0', '720.0']
'''

CONTEXT = "Youtube Video Transcoding"
TARGET = CuratedTarget(raw_name="utime", new_name="Transcription Time", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="i", new_name="Number of I Frames in the Video"),
            CuratedFeature(raw_name="p", new_name="Number of P Frames in the Video"),
            CuratedFeature(raw_name="b", new_name="Number of B Frames in the Video"),
            CuratedFeature(raw_name="i_size", new_name="Total Size in Bytes of I Videos"),
            CuratedFeature(raw_name="p_size", new_name="Total Size in Bytes of P Videos"),
            CuratedFeature(raw_name="o_codec", new_name="Output Codec Used for Transcoding"),
            CuratedFeature(raw_name="o_bitrate", new_name="Output Bitrate Used for Transcoding"),
            CuratedFeature(raw_name="o_framerate", new_name="Output Framerate Used for Transcoding"),
            CuratedFeature(raw_name="o_width", new_name="Output Width in Pixels Used for Transcoding"),
            CuratedFeature(raw_name="o_height", new_name="Output Height in Pixels Used for Transcoding")]