import pynput
import time

mouse = pynput.mouse.Controller()
def move_smooth(xm, ym, t):
    for i in range(t):
        if i < t/2:
            h = i
        else:
            h = t - i
        mouse.move(h*xm, h*ym)
        time.sleep(1/60)

move_smooth(0.5, 0, 200)
