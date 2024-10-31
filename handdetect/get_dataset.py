

from roboflow import Roboflow
rf = Roboflow(api_key="qoKjvPja7OXhxd8QijxP")
project = rf.workspace("ws-hgdem").project("gesture-zgba9")
version = project.version(1)
dataset = version.download("yolov8")
                