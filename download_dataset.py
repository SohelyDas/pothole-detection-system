from roboflow import Roboflow
rf = Roboflow(api_key="m5hvKVQAcIaplYg8D9QS")
project = rf.workspace("sohely-das").project("pothole-detection-j4qpn-mcodm")
version = project.version(1)
dataset = version.download("yolov8")