from ultralytics import YOLO
import time, os
import importlib

supported_sizes = importlib.import_module('mmcensor.const').supported_sizes

t_1 = time.perf_counter()
model = YOLO( '../neuralnet_models/640m.pt' )
for size in supported_sizes:
    model.export( format='engine',imgsz=size,half=True,dynamic=True)
    os.rename( '../neuralnet_models/640m.engine', '../neuralnet_models/640m-%d.engine'%size )
t_2 = time.perf_counter()
print( t_2-t_1 )
