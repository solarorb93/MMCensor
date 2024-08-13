import importlib
import tkinter as tk
import cv2
from mmcensor.decorate.decorator_utils import feature_selector
import mmcensor.geo as geo
import math

class decorator:

    def initialize( self, known_classes ):
        self.known_classes = known_classes
        self.classes = []
        self.strength=10

    def decorate( self, img, boxes ):
        condensed = geo.condense_boxes_single( boxes )

        for feature in condensed:
            if self.known_classes[feature] in self.classes:
                for box in condensed[feature]:
                    w = box[4]-box[2]
                    h = box[5]-box[3]
                    factor = 2 * math.ceil( self.strength * min( w,h ) / 100 /2 ) + 1
                    img[box[3]:box[5],box[2]:box[4]]=cv2.blur( img[box[3]:box[5],box[2]:box[4]], (factor,factor), cv2.BORDER_DEFAULT )

        return img

    def export_settings( self ):
        return( { 'classes': self.classes, 'strength': self.strength } )

    def import_settings( self, settings ):
        self.classes = settings['classes']
        self.strength = settings['strength']

    def short_desc( self ):
        return '%d classes, strength %d'%(len(self.classes),self.strength)

    def populate_config_frame( self, frame ):
        #self.feature_selector = importlib.import_module('decorator_utils').feature_selector()
        self.strength_var = tk.IntVar()

        tk.Label( frame, text="Strength (1 to 50 or higher):").grid(row=1,column=0,columnspan=3)
        self.strength_entry = tk.Entry( frame, textvariable=self.strength_var )
        self.strength_entry.delete(0,tk.END)
        self.strength_entry.insert(0,str(self.strength))
        self.strength_entry.grid(row=1,column=3)

        self.feature_selector = feature_selector()

        class_frame = tk.Frame(frame)
        self.feature_selector.populate_frame(class_frame, self.known_classes, self.classes)
        class_frame.grid(row=2,column=0,columnspan=4)

    def apply_config_from_config_frame( self ):
        self.classes = self.feature_selector.get_selected_classes()
        self.strength = self.strength_var.get()

    def destroy_config_frame( self ):
        return 0

