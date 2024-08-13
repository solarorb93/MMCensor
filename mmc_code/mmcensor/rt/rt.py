import numpy as np
import dxcam
import ctypes
import win32api, win32con, win32ui, win32gui
import time
import mmcensor.const as mmc_const
import mmcensor.geo as geo
import random
from multiprocessing import shared_memory, Manager, Process
import cv2
import threading
import os
import sys
import copy
import importlib
from functools import partial
import mmcensor.config as mmc_config
import mmcensor.nn as nn
import statistics
from datetime import datetime

def get_dxcams():
    # this is in a function because the weird backdoor
    # I do doesn't work inside a class
    # all of this is horrible and dxcam should just expose
    # the data I need.
    # I apologize for all of this
    cams = []
    outputs = dxcam.__factory.outputs
    for i in range(len(outputs)):
        for j in range(len( outputs[i] )):
            cam = dxcam.create( device_idx=i, output_idx=j, output_color='BGR' )
            cam_coords = cam._output.desc.DesktopCoordinates
            cams.append( { 'cam':cam, 'cam_coords': [ cam_coords.left, cam_coords.top, cam_coords.right, cam_coords.bottom ], 'desc':str(outputs[i][j]) } )
    return( cams )

class mmc_screencap:

    def initialize( self ):
        # determine screen geometry
        self.populate_dxcams()
        self.visible_bounds = self.get_visible_bounds()
        self.img_shape = ( self.visible_bounds[3] - self.visible_bounds[1], self.visible_bounds[2] - self.visible_bounds[0], 3 )

        # set up shared memory
        self.img_shm_name    = 'img_shm_name_%d'%random.randint(0,10000000)     # the actual image data
        self.img_coords_name = 'img_coords_name_%d'%random.randint(0,10000000)  # the coordinates of each window and its hwnd
        self.img_ref_name    = 'img_time_name_%d'%random.randint(0,100000000)   # the time of the snap, and the number of windows snapped

        self.img_shm = shared_memory.SharedMemory( name=self.img_shm_name, create=True, size = self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        self.img_coords_shm = shared_memory.SharedMemory( name = self.img_coords_name, create=True, size = 10000 )
        self.img_ref_shm = shared_memory.SharedMemory( name = self.img_ref_name, create=True, size = 1000 )

        self.img_shared = np.ndarray( self.img_shape, dtype = np.uint8, buffer = self.img_shm.buf        )
        self.img_coords = np.ndarray( (100, 5 ),      dtype = np.int64, buffer = self.img_coords_shm.buf )
        self.img_ref    = np.ndarray( ( 2, ),         dtype = np.int64, buffer = self.img_ref_shm.buf    )

        # warm up cams
        for cam in self.cams:
            _dummy = cam['cam'].grab()

    def populate_dxcams( self ):
        self.cams = get_dxcams()
        
    def get_visible_bounds( self ):
        l = t = r = b = None
        for cam in self.cams:
            l = cam['cam_coords'][0] if l is None else min( l, cam['cam_coords'][0] )
            t = cam['cam_coords'][1] if t is None else min( l, cam['cam_coords'][1] )
            r = cam['cam_coords'][2] if r is None else max( r, cam['cam_coords'][2] )
            b = cam['cam_coords'][3] if b is None else max( b, cam['cam_coords'][3] )

        return( [ l, t, r, b ] )

    def get_hwnds( self ):
        l = []

        for i in range(len(self.cams)):
            l.append( [ -1 * i, self.cams[i]['desc'] ] )

        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText( hwnd )
                if len(title):
                    l.append( [ hwnd, win32gui.GetWindowText(hwnd) ] )
        win32gui.EnumWindows(winEnumHandler, l)

        return( l )

    def get_hwnd_coords_unintersected( self, hwnd ):
        if hwnd <= 0:
            return self.cams[ -1 * hwnd ]['cam_coords']
        else:
            rect = ctypes.wintypes.RECT()
            DWMWA_EXTENDED_FRAME_BOUNDS = 9 # magic windows number 
            ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(hwnd),
              ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
              ctypes.byref(rect),
              ctypes.sizeof(rect)
              )

            window_coords = [ rect.left, rect.top, rect.right, rect.bottom ]
            return( window_coords )

    def get_hwnd_coords( self, hwnd ):
            window_coords = self.get_hwnd_coords_unintersected( hwnd )
            visible_coords = geo.intersection_box( window_coords, self.visible_bounds )

            return visible_coords

    def snap_hwnds( self, hwnds ):
        tasks = []

        hwnd_coords = {}
        cam_tasks = [ None for cam in self.cams ]

        for hwnd in hwnds:
            this_hwnd_coords = self.get_hwnd_coords( hwnd )
            if this_hwnd_coords is not None:
                for i in range(len(self.cams)):
                    cam = self.cams[i]
                    int_xyxy = geo.intersection_box( cam['cam_coords'], this_hwnd_coords )
                    if int_xyxy is not None:
                        hwnd_coords[hwnd] = this_hwnd_coords # we found at least one part in a visible area
                        grab_coords = [
                                int_xyxy[0]-cam['cam_coords'][0],
                                int_xyxy[1]-cam['cam_coords'][1],
                                int_xyxy[2]-cam['cam_coords'][0],
                                int_xyxy[3]-cam['cam_coords'][1],
                                ]
                        if cam_tasks[i] is not None:
                            cam_tasks[i] = geo.union_box( cam_tasks[i], grab_coords )
                        else:
                            cam_tasks[i] = grab_coords

        #self.profiler.mark( 'get_coords' )

        snap_time = time.perf_counter_ns()

        for i in range(len( cam_tasks )):
            if cam_tasks[i] is not None:
                cam = self.cams[i]
                subimg = cam['cam'].grab( region=tuple(cam_tasks[i] ) )
                if subimg is not None:
                    subimg_xyxy = [
                            cam_tasks[i][0]+cam['cam_coords'][0]-self.visible_bounds[0],
                            cam_tasks[i][1]+cam['cam_coords'][1]-self.visible_bounds[1],
                            cam_tasks[i][2]+cam['cam_coords'][0]-self.visible_bounds[0],
                            cam_tasks[i][3]+cam['cam_coords'][1]-self.visible_bounds[1],
                            ]
                    self.img_shared[subimg_xyxy[1]:subimg_xyxy[3],subimg_xyxy[0]:subimg_xyxy[2]]=subimg

        i = 0
        for hwnd in hwnd_coords:
            self.img_coords[i] = ( 
                    hwnd_coords[hwnd][0]-self.visible_bounds[0], 
                    hwnd_coords[hwnd][1]-self.visible_bounds[1], 
                    hwnd_coords[hwnd][2]-self.visible_bounds[0], 
                    hwnd_coords[hwnd][3]-self.visible_bounds[1], 
                    hwnd
                    )
            i = i + 1
            #self.profiler.mark('shared_coords')

        self.img_ref[0] = snap_time
        self.img_ref[1] = len(hwnd_coords)

        #self.profiler.mark('set_img_ref')

# this class is created on the main process
# see mmc_detect_remote_func and 
# mmc_detect_loop_remote for the class
# that is created on the remote process
class mmc_detect_loop_async:

    def initialize( self, img_shm_name, img_coords_name, img_ref_name, img_shape, sizes, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name ):
        manager = Manager()
        self.sizes = manager.list()
        self.state = manager.list()
        self.state.append( 0 )
        self.sizes.extend( sizes )
        self.P1 = Process( target = mmc_detect_loop_remote, args = ( self.sizes, self.state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name ) )

    def start( self ):
        self.P1.start()

        while( self.state[0] == 0 and self.P1.is_alive() ):
            print( "waiting for neural net to be ready..." )
            time.sleep( 2 )

    def shutdown( self ):
        if self.P1.is_alive():
            self.P1.terminate()
            self.P1.join()

def mmc_detect_loop_remote( sizes, state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name):
    detector = mmc_detect_loop_class()
    detector.initialize( sizes, state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name )
    detector.go_detect()

class mmc_detect_loop_class:

    def initialize( self, sizes, state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name ):
        from ultralytics import YOLO
        self.sizes = sizes
        self.state = state
        self.env = os.getenv( 'mmcNNenv' )
        self.last_t = 0
        self.fps_limit = 300
        self.last_detect_finish = 0

        self.working_sizes = mmc_const.supported_sizes
        if self.env == 'openvino':
            self.model = YOLO( "../neuralnet_models/640m_openvino_model" )
            self.single_model = True
        elif self.env == 'directml':
            self.model = YOLO( "../neuralnet_models/640m.onnx" )
            self.single_model = True
        elif self.env == 'tensorrt':
            self.models = {}
            self.working_sizes = []
            for size in mmc_const.supported_sizes:
                engine_path = "../neuralnet_models/640m-%d.engine"%size
                if os.path.isfile( engine_path ):
                    self.models[ size ] = YOLO( engine_path )
                    self.working_sizes.append( size )
            self.single_model = False
        else:
            self.model = YOLO( "../neuralnet_models/640m.pt" )
            self.single_model = True

        warmup_img = np.full( ( 2560, 2560, 3 ), 127, dtype=np.uint8 )
        for size in self.working_sizes:
            model = self.get_model_for_size( size )
            if model is not None:
                self.get_model_for_size(size).predict(warmup_img, imgsz=size, verbose=False )

        self.img_shape = img_shape

        self.img_shm = shared_memory.SharedMemory( name=img_shm_name )
        self.img_coords_shm = shared_memory.SharedMemory( name = img_coords_name )
        self.img_ref_shm = shared_memory.SharedMemory( name = img_ref_name )

        self.img_shared = np.ndarray( self.img_shape, dtype = np.uint8, buffer = self.img_shm.buf        )
        self.img_coords = np.ndarray( (100, 5 ),      dtype = np.int64, buffer = self.img_coords_shm.buf )
        self.img_ref    = np.ndarray( ( 2, ),         dtype = np.int64, buffer = self.img_ref_shm.buf    )

        self.boxes_shm = shared_memory.SharedMemory( name=boxes_shm_name )
        self.box_hwnds_shm = shared_memory.SharedMemory( name=box_hwnds_shm_name )
        self.box_info_shm = shared_memory.SharedMemory( name = box_info_shm_name )

        self.boxes_np = np.ndarray( (20,500,8), dtype = np.int64, buffer = self.boxes_shm.buf        )
        self.box_hwnds_np = np.ndarray( (50,4), dtype = np.int64, buffer = self.box_hwnds_shm.buf        )
        self.box_info_np = np.ndarray( (4,),      dtype = np.int64, buffer = self.box_info_shm.buf )

        self.state[0] = 1

    def get_model_for_size( self, size ):
        if self.single_model:
            return self.model
        else:
            return self.models[ size ]

    def go_detect( self ):
        n = 0
        t_start = time.perf_counter()
        self.profiler = profiler()
        self.profiler.initialize( 5, 0.0001 )
        while( True ):
            self.profiler.loop()
            self.profiler.mark( "start" )
            sstime = self.img_ref[0]
            self.profiler.mark( "got_time" )

            if sstime > self.last_t:
                num_hwnds = self.img_ref[1]
                self.profiler.mark( "got_hwnds" )

                if num_hwnds:
                    batch = []
                    hwnds = []
                    outs = {}
                    for i in range( num_hwnds ):
                        hwnds.append( self.img_coords[i][4] )
                        self.profiler.mark( "got_hwnd" )
                        batch.append( np.ascontiguousarray(self.img_shared[self.img_coords[i][1]:self.img_coords[i][3],self.img_coords[i][0]:self.img_coords[i][2]]) )
                        self.profiler.mark( "got_batch" )
                        outs[ self.img_coords[i][4] ] = {}

                    self.profiler.mark( "presizes" )
                    for size in self.sizes:
                        if self.env == 'tensorrt': # tensorrt needs to have engine files designed for batching
                            output = [ self.get_model_for_size(size).predict( x, imgsz=size, verbose = False )[0] for x in batch ]
                        else:
                            output = self.get_model_for_size(size).predict( batch, imgsz=size, verbose = False )
                        time.sleep(0.6)
                        #if random.randint(0,100) <2:
                            #raise Exception( "test throw" )
                        self.profiler.mark( "predict" )
                        for i in range( num_hwnds ):
                            outs[ hwnds[ i ] ][size] = output[i].boxes.cpu().numpy()
                            self.profiler.mark( "append_out" )

                    self.profiler.mark( "done_predict" )
                    self.box_info_np[0] = sstime
                    i=0
                    for hwnd in outs:
                        j=0
                        for size in outs[hwnd]:
                            for box in outs[hwnd][size]:
                                self.boxes_np[i][j] = (sstime,box.cls[0].item(),box.xyxy[0][0].item(),box.xyxy[0][1].item(),box.xyxy[0][2].item(),box.xyxy[0][3].item(),1,size)
                                j = j+1
                        self.profiler.mark( "copied_box" )
                        self.box_hwnds_np[i]=(hwnd,j,self.img_coords[i][2]-self.img_coords[i][0],self.img_coords[i][3]-self.img_coords[i][1])
                        self.profiler.mark( "wrote_hwnds" )
                        i=i+1
                    self.box_info_np[1] = i
                    self.box_info_np[2] = nn.sizes_to_key( self.sizes )
                    self.box_info_np[3] = sstime

                    self.profiler.mark( "done_outs" )

                self.last_t = sstime

            fps_limit_sleep = 1/self.fps_limit - ( time.perf_counter() - self.last_detect_finish )
            if fps_limit_sleep > 0:
                time.sleep( fps_limit_sleep )
            self.profiler.mark( "done_sleep" )

            self.last_detect_finish = time.perf_counter()

            n = n+1
            if n == 100:
                t_end = time.perf_counter()
                print( "100 detections in %.2f seconds, or %.1ffps"%(t_end - t_start, 100 / ( t_end - t_start ) ) )
                t_start = t_end
                n = 0
            self.profiler.mark( "done" )

class profiler:
    times = {}
    n = 0
    last = 0

    def initialize( self, report_freq, time_threshold ):
        self.n = 0
        self.times = {}
        self.report_start = time.perf_counter()
        self.report_freq = report_freq
        self.last = time.perf_counter()
        self.loop_start = time.perf_counter()
        self.time_threshold = time_threshold

    def mark( self, label ):
        new = time.perf_counter()
        elapsed = new - self.last
        self.last = new

        if label in self.times:
            if label in self.seen_in_loop:
                self.times[label][-1] = self.times[label][-1] + elapsed
            else:
                self.times[label].append( elapsed )
                self.seen_in_loop[label]=None
        else:
            self.times[label] = [ elapsed ]

    def loop( self ):
        self.n = self.n + 1
        if time.perf_counter() - self.report_start > self.report_freq:
            label_sum = {}
            for label in self.times:
                label_sum[ label ] = sum( self.times[label] )
                if max(self.times[label]) > self.time_threshold:
                    print( label.ljust(15), 'avg %.1fms max %.1fms std %.1fms count %d/%d'%(
                        label_sum[label]/self.n*1000,
                        max(self.times[label])*1000,
                        statistics.stdev( self.times[label] )*1000 if len(self.times[label])>1 else 0,
                        len(self.times[label]),
                        self.n
                        ), '%.1f fps'%(self.n/(time.perf_counter()-self.report_start),) if label=='__loop__' else '' )

            self.times = {}
            self.report_start = time.perf_counter()
            self.loop_start = time.perf_counter()
            self.n = 0

        else:
            if '__loop__' in self.times:
                self.times['__loop__'].append( time.perf_counter() - self.loop_start )
            else:
                self.times['__loop__'] = [ time.perf_counter() - self.loop_start ]
            self.loop_start = time.perf_counter()

        self.seen_in_loop = {}

class mmc_realtime:

    def initialize( self ):
        self.sc = mmc_screencap()
        self.sc.initialize()
        self.ready = False
        self.running = False
        self.hwnds = []
        self.cv_title_template = 'mmcensor-%d-%%d'%random.randint(0,100000 )
        self.profiler = profiler()
        self.profiler.initialize( 5, 0.0001 )
        self.sc.profiler = self.profiler
        self.threaded_screenshot = True
        self.decorators = []
        self.open_windows = {}

        self.size_detection_timings = {}
        self.size_delays = {}

        # set up shared memory
        self.boxes_shm_name    = 'boxes_shm_name_%d'%random.randint(0,10000000)     # [ [ t, cls, x1, y1, x2, y2, prob, size ] ]
        self.box_hwnds_shm_name    = 'box_hwnds_shm_name_%d'%random.randint(0,10000000)     # [ [ hwnd, numboxes ] ]
        self.box_info_shm_name = 'box_info_shm_name_%d'%random.randint(0,10000000)  # [ t, numhwnds, t ]

        self.boxes_shm = shared_memory.SharedMemory( name=self.boxes_shm_name, create=True, size = 8 * 20 * 9 * 500 ) # 8 bytes times twenty hwnds times nine fields times 500 boxes
        self.box_hwnds_shm = shared_memory.SharedMemory( name=self.box_hwnds_shm_name, create=True, size = 10000 ) 
        self.box_info_shm = shared_memory.SharedMemory( name = self.box_info_shm_name, create=True, size = 10000 )

        self.boxes_np = np.ndarray( (20,500,8), dtype = np.int64, buffer = self.boxes_shm.buf        )
        self.box_hwnds_np = np.ndarray( ( 50, 4), dtype = np.int64, buffer = self.box_hwnds_shm.buf        )
        self.box_info_np = np.ndarray( (4,),      dtype = np.int64, buffer = self.box_info_shm.buf )
        self.box_hwnds_np[:][:]=0

        self.boxes = np.ndarray( (50,20000,8), dtype=np.int64 )
        self.boxes_hwnd_index = {}
        self.hwnd_times = {} # hwnd: [ t, first_index, last_index ]
        self.last_detection_found = 0

        self.detector_async = mmc_detect_loop_async()
        self.detector_async.initialize( self.sc.img_shm_name, self.sc.img_coords_name, self.sc.img_ref_name, self.sc.img_shape, [], self.boxes_shm_name, self.box_hwnds_shm_name, self.box_info_shm_name )
        self.sizes = self.detector_async.sizes
        self.to_show = {}

        self.on_gray_callback = None
        self.off_gray_callback = None
        self.gray_state = False

    def take_screenshot( self ):
        for hwnd in self.to_show:
            if self.to_show[hwnd] is not None:
                now = datetime.today().strftime('%Y%m%d%H%M%S%f')
                cv2.imwrite( '../screenshots/%s.jpg'%now, self.to_show[hwnd] )

    def update_sizes( self, sizes ):
        while( len( self.sizes ) ):
            self.sizes.pop(0)
        self.sizes.extend( sizes )

    def make_ready( self ):
        self.time_safety_ns = mmc_config.get_time_settings()['time-safety'] * 1000 * 1000 * 1000
        self.detector_async.start()
        self.ready = True

    def go_decorate( self ):
        if not self.ready:
            return

        self.running = True
        img_buffer = []
        self.hwnd_pos = {}

        n = 0
        t_fps = time.perf_counter()

        self.delay_key_print_history = {}

        while( True ):
            self.profiler.loop()

            if not self.detector_async.P1.is_alive():
                print( "DETECTOR THREAD FAILED.  EXITING.  PLEASE REPORT ANY ERRORS PRINTED ABOVE." )
                sys.exit()

            t1 = threading.Thread( target=self.sc.snap_hwnds, args = [ self.hwnds ] )
            if self.threaded_screenshot:
                t1.start()
            else:
                t1.run()

            self.profiler.mark('after_snap')

            num_snapped = self.sc.img_ref[1]
            t_snapped = self.sc.img_ref[0]

            self.profiler.mark('got_ref')

            coords = self.sc.img_coords.copy()

            self.profiler.mark('copied_coords')

            # grab the snapped images
            img_collection = {}
            for i in range(num_snapped):
                hwnd = coords[i][4]
                img_collection[ hwnd ] = [
                    self.sc.img_shared[coords[i][1]:coords[i][3],coords[i][0]:coords[i][2]].copy(), 
                    [ coords[i][0], coords[i][1], coords[i][2], coords[i][3] ] 
                    ]

            self.profiler.mark( 'copied_img' )

            img_buffer.append( [ t_snapped, img_collection ] )

            self.profiler.mark( 'appended_buffer' )

            delay_key = ( len( self.hwnds ), nn.sizes_to_key( self.sizes ) )
            if delay_key in self.size_delays:
                delay = self.size_delays[ delay_key ]
            else:
                if delay_key in self.size_detection_timings:
                    if len( self.size_detection_timings[ delay_key ] ) > 15 or ( len(self.size_detection_timings[delay_key] ) > 4 and sum( self.size_detection_timings[delay_key] ) > 4 * 1000000000 ):
                        print( self.size_detection_timings[ delay_key ] )
                        delay = 2.2 * sum( self.size_detection_timings[delay_key] ) / len( self.size_detection_timings[delay_key] ) + self.time_safety_ns/2
                        self.size_delays[delay_key] = delay
                        print( 'delay set to %.3fs'%(delay/1000000000,) )
                    else:
                        if delay_key not in self.delay_key_print_history or len(self.delay_key_print_history[ delay_key ]) != len(self.size_detection_timings[ delay_key ]):
                            print( "calculating delay....", self.size_detection_timings[delay_key] )
                            self.delay_key_print_history[ delay_key ] = self.size_detection_timings[ delay_key ].copy()
                            delay = 3*1000000000
                else:
                    if delay_key not in self.delay_key_print_history or self.delay_key_print_history[ delay_key ] != []:
                        self.delay_key_print_history[ delay_key ] = []
                        print( "calculating delay...." )
                        delay = 3*1000000000

            oldest_keep_img = time.perf_counter_ns() - delay

            # eliminate old images
            while( len( img_buffer ) > 1 and img_buffer[1][0] < oldest_keep_img ):
                img_buffer.pop(0)

            self.profiler.mark( 'popped_old' )

            # eliminate old detections
            to_show_time_ns = img_buffer[0][0]
            oldest_detection = to_show_time_ns - self.time_safety_ns
            latest_detection = to_show_time_ns + self.time_safety_ns
            for hwnd in self.hwnd_times:
                popped = False
                while( len( self.hwnd_times[hwnd] )>1 and self.hwnd_times[hwnd][1][0] < oldest_detection ):
                    self.hwnd_times[hwnd].pop(0)
                    popped = True
                if popped:
                    index = self.boxes_hwnd_index[hwnd]
                    first_index_to_keep = self.hwnd_times[hwnd][0][1]
                    last_index_to_keep = self.hwnd_times[hwnd][-1][2]
                    self.boxes[index][:last_index_to_keep-first_index_to_keep+1]=self.boxes[index][first_index_to_keep:last_index_to_keep+1]
                    for elt in self.hwnd_times[hwnd]:
                        elt[1] = elt[1] - first_index_to_keep
                        elt[2] = elt[2] - first_index_to_keep

            self.profiler.mark( 'copied_Q' )

            detection_time = self.box_info_np[0]
            if detection_time > self.last_detection_found:
                for i in range(self.box_info_np[1]):
                    hwnd = self.box_hwnds_np[i][0]
                    num_boxes = self.box_hwnds_np[i][1]
                    if hwnd not in self.boxes_hwnd_index:
                        self.boxes_hwnd_index[hwnd] = len(self.boxes_hwnd_index)
                        self.hwnd_times[ hwnd ] = []
                    if len( self.hwnd_times[hwnd] ):
                        new_first_index = self.hwnd_times[hwnd][-1][2] + 1
                        new_last_index = self.hwnd_times[hwnd][-1][2] + num_boxes
                    else:
                        new_first_index = 0
                        new_last_index = num_boxes-1
                    if num_boxes:
                        self.boxes[self.boxes_hwnd_index[hwnd]][new_first_index:new_last_index+1]=self.boxes_np[i][0:num_boxes]
                    self.hwnd_times[hwnd].append( [ detection_time, new_first_index, new_last_index ] )
                    if self.last_detection_found > 0:
                        detected_delay_key=(self.box_info_np[1],self.box_info_np[2])
                        if detected_delay_key not in self.size_delays:
                            self.size_detection_timings.setdefault(detected_delay_key,[]).append( detection_time - self.last_detection_found )
                self.last_detection_found = detection_time

            self.profiler.mark( 'reshaped_boxes' )

            has_gray_img = False
            for hwnd in self.to_show:
                if hwnd not in img_buffer[-1][1]:
                    self.to_show[hwnd] = None
            for hwnd in img_buffer[-1][1]: # the list of things to show is whatever the *latest* list of captured coordinates is
                self.profiler.mark( 'pre_full' )
                new_xyxy = img_buffer[-1][1][hwnd][1]
                self.to_show[ hwnd ] = np.full( (new_xyxy[3]-new_xyxy[1], new_xyxy[2]-new_xyxy[0], 3 ), 127, dtype=np.uint8 )
                self.profiler.mark( 'post_full' )

                if hwnd in img_buffer[0][1] and hwnd in self.hwnd_times and self.hwnd_times[hwnd][0][0] < img_buffer[0][0] and self.hwnd_times[hwnd][-1][0] > img_buffer[0][0]:
                    old_xyxy = img_buffer[0][1][hwnd][1]
                    self.profiler.mark( 'got_old_xyxy' )
                    min_h = min( old_xyxy[3] - old_xyxy[1], new_xyxy[3] - new_xyxy[1] )
                    min_w = min( old_xyxy[2] - old_xyxy[0], new_xyxy[2] - new_xyxy[0] )

                    self.to_show[hwnd][0:min_h,0:min_w] = img_buffer[0][1][hwnd][0][0:min_h,0:min_w]
                    self.profiler.mark( 'populated_show' )

                    for i in range(len(self.hwnd_times[hwnd])):
                        if self.hwnd_times[hwnd][i][0]>latest_detection:
                            break
                    last_box_index = self.hwnd_times[hwnd][i][2]

                    # you could do this faster by intersecting with window size as you
                    # go, but it's really annoying
                    # this probably isn't that slow
                    relevant_boxes = self.boxes[self.boxes_hwnd_index[hwnd]][0:last_box_index+1].copy()
                    relevant_boxes = relevant_boxes[np.less(relevant_boxes[:,2],min_w )] #discard boxes that start to the right of min_w
                    relevant_boxes = relevant_boxes[np.less(relevant_boxes[:,3],min_h )]
                    relevant_boxes[:,4]=np.fmin(relevant_boxes[:,4],min_w)
                    relevant_boxes[:,5]=np.fmin(relevant_boxes[:,5],min_h)

                    for decorator in self.decorators:
                        self.to_show[hwnd][0:min_h,0:min_w] = decorator.decorate( self.to_show[hwnd][0:min_h,0:min_w], relevant_boxes )

                    self.profiler.mark( 'decorated' )
                else:
                    has_gray_img = True

                self.show( self.to_show[ hwnd ], hwnd, new_xyxy )
                self.profiler.mark( 'showed' )

            # avoid deleting from dict while iterating over dict
            windows_to_close = []
            for window_hwnd in self.open_windows:
                if window_hwnd not in self.to_show or self.to_show[window_hwnd] is None:
                    windows_to_close.append( window_hwnd )
            for window_hwnd in windows_to_close:
                cv2.destroyWindow( self.open_windows[window_hwnd] )
                del self.open_windows[ window_hwnd ]
                del self.hwnd_pos[ window_hwnd ]

            if self.gray_state == True and has_gray_img == False and self.off_gray_callback is not None:
                self.off_gray_callback()
                self.gray_state = False

            if self.gray_state == False and has_gray_img == True and self.on_gray_callback is not None:
                self.on_gray_callback()
                self.gray_state = True

            self.profiler.mark('closed_windows')

            n = n+1
            if n == 100:
                elapsed = time.perf_counter() - t_fps
                print( '100 frames in %.3fs, or %.1fps'%( elapsed, 100/elapsed ))
                n = 0
                t_fps = time.perf_counter()

            self.profiler.mark( 'post_fps' )

            if( cv2.waitKey(1) == ord('q') or self.running == False ):
                cv2.destroyAllWindows()
                self.open_windows = {}
                if t1.is_alive():
                    t1.join()
                break

            self.profiler.mark( 'post_wait' )

            if t1.is_alive():
                t1.join()

            self.profiler.mark( 'post_join' )

    def show( self, img, real_hwnd, new_xyxy ):
        cv_title = self.cv_title_template%real_hwnd
        self.open_windows[ real_hwnd ] = cv_title
        cv2.imshow( cv_title, img )
        #cv2.imshow( 'rec', img )
        self.profiler.mark( 'show_call')
        if real_hwnd not in self.hwnd_pos or self.hwnd_pos[real_hwnd] != new_xyxy:
            hwnd = win32gui.FindWindow(None, cv_title )
            self.profiler.mark( 'show_find_hwnd')
            if hwnd:
                # Get window style and perform a 'bitwise or' operation to make the style layered and transparent, achieving
                # the clickthrough property
                ctypes.windll.user32.SetWindowDisplayAffinity( hwnd, 0x00000011 )
                self.profiler.mark( 'show_affinity')
                l_ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                self.profiler.mark( 'show_ex_style')
                l_ex_style |= win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, l_ex_style)
                self.profiler.mark( 'show_set_long')

                # Set the window to be transparent and appear always on top
                win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 255, win32con.LWA_ALPHA)  # transparent
                self.profiler.mark( 'show_transparent')
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, self.sc.visible_bounds[0] + new_xyxy[0], self.sc.visible_bounds[1] + new_xyxy[1], new_xyxy[2]-new_xyxy[0], new_xyxy[3]-new_xyxy[1], 0 )
                self.profiler.mark( 'show_pos1')
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOP,     self.sc.visible_bounds[0] + new_xyxy[0], self.sc.visible_bounds[1] + new_xyxy[1], new_xyxy[2]-new_xyxy[0], new_xyxy[3]-new_xyxy[1], 0 )
                self.profiler.mark( 'show_pos2')

                GWL_STYLE = -16

                currentStyle = win32gui.GetWindowLong(hwnd, GWL_STYLE)
                self.profiler.mark( 'show_getlonggwl')

                #  remove titlebar elements
                currentStyle = currentStyle & ~(0x00C00000)  #  WS_CAPTION
                currentStyle = currentStyle & ~(0x00080000)  #  WS_SYSMENU
                currentStyle = currentStyle & ~(0x00040000)  #  WS_THICKFRAME
                currentStyle = currentStyle & ~(0x20000000)  #  WS_MINIMIZE
                currentStyle = currentStyle & ~(0x00010000)  #  WS_MAXIMIZEBOX

                #  apply new style
                win32gui.SetWindowLong(hwnd, GWL_STYLE, currentStyle)
                self.profiler.mark( 'show_setlonggwl')
                self.hwnd_pos[ real_hwnd ] = new_xyxy
                
