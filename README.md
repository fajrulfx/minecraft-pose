# minecraft-pose
A simple program to control Minecraft using body pose. Inspired by Fundy—light version. 

While Fundy use an advanced pose detection that require some data trainings, this program simply just detect pose landmarks and use the inputs to control minecraft.

# Youtube video
<a href="https://www.youtube.com/watch?v=fcOwzUq6cvs"><img src="https://i.ytimg.com/vi/fcOwzUq6cvs/maxresdefault.jpg" width="400"/></a>

# Dependencies
1. Python
2. Library:
   - mediapipe
   - open-cv
   - pydirectinput

# How to use
1. Install Python
2. Install `pip` (if you haven't)
3. Install libraries:
   - `pip install mediapipe`
   - `pip install opencv-python`
   - `pip install pydirectinput`
4. Open Minecraft, and run the program `python pose.py`
5. In Minecraft, you need to change mouse setting: Option > Control > Mouse Settings > Raw Input > OFF 
