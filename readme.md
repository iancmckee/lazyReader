# Thanks for checking out the Repo!
## A lot of inspiration came from the following repo detecting head nodding/shaking 
  - https://gist.github.com/smeschke/e59a9f5a40f0b0ed73305d34695d916b

## There are some built in commands that can customize how much you have to look up/ look down to trigger scroll
  - CTRL+ALT removes existing trigger limites (defaults to 35)
  - CTRL+F1 this sets the lookup/ scroll down limit to the y-movement you're currently at
  - CTRL+F2 this sets the look down/ scroll up limit to the y-movement you're currently at
  - ESC will reinitialize your face. if you move around a lot it loses where your face it. if you don't see the dot in between your eyes i'd click esc

### Things of note - 
  - We're using a fixed videocapture, it's just looking for whatever 0 index webcam you have, if it's not connecting to the right webcam, unplug your other webcams. You can also change line 16 `cap = cv2.VideoCapture(0)` and cycle through integers for cameras
