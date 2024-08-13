from ultralytics import YOLO
import time, os

t_1 = time.perf_counter()
model = YOLO( '../neuralnet_models/640m.pt' )
model.export( format='openvino', half=False, dynamic=True )

t_2 = time.perf_counter()
print( t_2-t_1 )

