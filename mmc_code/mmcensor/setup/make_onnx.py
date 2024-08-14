from ultralytics import YOLO
import time, os

t_1 = time.perf_counter()
model = YOLO( '../neuralnet_models/640m.pt' )
model.export( format='onnx', dynamic=True )

# someday, it would be nice to get half=True models for onnx for directml
# unfortunately, it seems these can only be generated with CUDA, which
# of course defeats the point, since we should be using CUDA, not directml
# it's possible that in the future I'll distribute onnx model files 
# generated on my machine.  In testing, the speed benefit is about 10%
# over just a single dynamic onnx file.
#import importlib
#supported_sizes = importlib.import_module( 'mmcensor.const' ).supported_sizes
#for size in supported_sizes:
    #model.export( format='onnx', imgsz=size, dynamic=False, half=True, simplify=True )
    #os.rename( '../neuralnet_models/640m.onnx', '../neuralnet_models/640m-%d.onnx'%size )
t_2 = time.perf_counter()
print( t_2-t_1 )

