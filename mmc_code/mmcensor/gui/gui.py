import tkinter as tk
from tkinter import ttk
from mmcensor.rt import mmc_realtime
import os
import sys
import importlib
import mmcensor.const as mmc_const
from functools import partial
import threading
import json

class mmc_gui:
    
    def initialize( self ):
        ##########################
        ## make root window
        ## add save and load buttons
        ## make tabs
        ##########################
        self.root = tk.Tk()
        self.root.protocol( "WM_DELETE_WINDOW", self.on_close )
        self.root.geometry( "800x800" )

        self.save_button = tk.Button( self.root, text= "Save", command = self.save_pushed )
        self.save_as_button = tk.Button( self.root, text = "Save As (not yet implemented)" )
        self.load_button = tk.Button( self.root, text = "Load", command=self.load_pushed )

        self.save_button.grid( row=0, column = 0 )
        self.save_as_button.grid( row=0, column=1 )
        self.load_button.grid( row = 0, column = 2 )

        tab_parent = ttk.Notebook( self.root )
        self.tab_decorate = ttk.Frame( tab_parent )
        self.tab_realtime = ttk.Frame( tab_parent )
        
        tab_parent.add( self.tab_decorate, text="Decorators" )
        tab_parent.add( self.tab_realtime, text="Realtime" )
        tab_parent.grid( row = 1, column = 0, columnspan = 4 )

        #############################
        ## make realtime tab
        #############################
        self.rt = mmc_realtime()
        self.rt.initialize()
        self.rt.on_gray_callback = self.up
        self.rt.off_gray_callback = self.down

        self.ready_button = tk.Button( self.tab_realtime, text = "Make Ready", command = self.make_ready_pushed                )
        self.start_button = tk.Button( self.tab_realtime, text = "Start",      command = self.start_pushed, state='disabled'   )
        self.stop_button  = tk.Button( self.tab_realtime, text = "Stop",       command = self.stop_pushed,  state='disabled'   )
        self.screenshot_button = tk.Button( self.tab_realtime, text = "Screenshot", command = self.screenshot_pushed, state='disabled' )

        self.ready_button.grid( row=0, column=0 )
        self.start_button.grid( row=0, column=1 )
        self.stop_button.grid(  row=0, column=2 )
        self.screenshot_button.grid( row=0, column=3 )

        # placeholder for realtime threads
        self.t_ready = None
        self.t_decorate = None

        self.known_hwnds = []

        self.get_hwnds_button = tk.Button( self.tab_realtime, text = "Refresh Window List", command = self.refresh_hwnds )
        self.get_hwnds_button.grid( row=0, column=4 )

        self.window_list = tk.Listbox( self.tab_realtime, selectmode='multiple', width=80, exportselection=0 )
        self.window_list.bind( '<<ListboxSelect>>', self.change_hwnds_selection )
        self.window_list.grid( row=1, column=0, columnspan = 5 )

        self.refresh_hwnds()

        self.size_checks = []
        for i in range(len(mmc_const.supported_sizes ) ):
            iv = tk.IntVar( value=(i<2) )
            tk.Checkbutton( self.tab_realtime, text='net size %s'%(mmc_const.supported_sizes[i],),onvalue=1,offvalue=0,variable=iv,command=self.update_sizes).grid(row=2+i,column=0)
            self.size_checks.append( iv )

        ################################
        ## make decorator tab
        ################################
        self.decorator_widgets = []
        self.known_decorators = self.get_known_decorators()
        self.decorator_types = []
        self.selected_new_decorator = tk.StringVar()

        self.new_decorator_combobox = ttk.Combobox( self.tab_decorate, textvariable=self.selected_new_decorator, values = self.known_decorators)
        self.new_decorator_combobox.current(0)
        self.add_decorator_button = tk.Button( self.tab_decorate,  text= "Add decorator", command = self.add_selected_decorator )
        self.decorators_frame = tk.Frame( self.tab_decorate )

        self.new_decorator_combobox.grid( row=0, column= 0 )
        self.add_decorator_button.grid( row=0, column = 1 )
        self.decorators_frame.grid( row=1, column=0, columnspan = 3 )
        self.decorator_config_frame = None
        self.decorator_being_configured = None

        #self.rt.decorators.append( importlib.import_module("decorators." + "bar" ).decorator() )
        #self.rt.decorators[0].import_settings( { 'color': (255,0,255), 'classes': [
            #'VULVA_COVERED',
            #'BUTTOCKS_EXPOSED',
            #'FEMME_BREAST_EXPOSED',
            #'VULVA_EXPOSED',
            #'ANUS_EXPOSED',
            #'ANUS_COVERED',
            #'BREAST_COVERED',
            #'BUTTOCKS_COVERED',
        #] } )

        self.load_pushed()
        self.update_sizes()

        self.root.mainloop()

    def up( self ):
        self.root.attributes( '-topmost', True )

    def down( self ):
        self.root.attributes( '-topmost', False )

    def update_sizes( self ):
        sizes = []
        for i in range(len(mmc_const.supported_sizes)):
            if self.size_checks[i].get():
                sizes.append( mmc_const.supported_sizes[i] )

        self.rt.update_sizes(sizes)

    def get_known_decorators( self ):
        paths = [ f.name for f in os.scandir('mmcensor/decorate') if f.is_dir() ]
        if '__pycache__' in paths:
            paths.remove( '__pycache__' )
        return( paths )

    def add_selected_decorator( self ):
        if len(self.selected_new_decorator.get() ):
            self.add_decorator( self.selected_new_decorator.get() )

    def add_decorator( self, decorator_type ):
        decorator = importlib.import_module( 'mmcensor.decorate.%s'%decorator_type ).decorator()
        decorator.initialize( mmc_const.nudenet_v3_classes )
        self.rt.decorators.append( decorator )
        self.decorator_types.append( decorator_type )
        self.redraw_decorators()

    def redraw_decorators( self ):
        for w in self.decorators_frame.winfo_children():
            w.destroy()

        self.decorator_being_configured = None

        for i in range(len(self.rt.decorators)):
            tk.Button( self.decorators_frame,text='x', command = partial( self.delete_decorator, i ) ).grid(row=i,column=0)
            tk.Label( self.decorators_frame, text=self.decorator_types[i] ).grid(row=i,column=1)
            tk.Label( self.decorators_frame, text=self.rt.decorators[i].short_desc() ).grid(row=i,column=2)
            tk.Button( self.decorators_frame, text='configure', command= partial( self.configure_decorator, i ) ).grid(row=i, column=3) 
            
    def delete_decorator( self, index ):
        self.rt.decorators.pop(index)
        self.decorator_types.pop(index)
        self.redraw_decorators()

    def configure_decorator( self, index ):
        self.redraw_decorators()
        self.decorator_config_frame = tk.Frame( self.tab_decorate )
        self.decorator_save_config_button = tk.Button( self.decorators_frame, text="apply config", command = partial( self.apply_decorator_config, index ))
        self.decorator_close_config_button = tk.Button( self.decorators_frame, text="close", command = partial( self.close_decorator_config, index ))
        self.decorator_save_config_button.grid(row=index, column=4 )
        self.decorator_close_config_button.grid(row=index, column=5 )
        self.rt.decorators[index].populate_config_frame( self.decorator_config_frame )
        self.decorator_config_frame.grid(row=1,column=4, columnspan=2, rowspan=30)

    def apply_decorator_config( self, index ):
        self.rt.decorators[index].apply_config_from_config_frame()

    def close_decorator_config( self, index ):
        self.rt.decorators[index].destroy_config_frame()
        self.decorator_config_frame.destroy()
        self.redraw_decorators()
    
    def save_pushed( self ):
        save_data = []
        for i in range( len( self.rt.decorators ) ):
            save_data.append( [ self.decorator_types[i], self.rt.decorators[i].export_settings() ] )

        with open('saved_settings.json', 'w') as f:
            json.dump( save_data, f )

    def load_pushed( self ):
        if not os.path.isfile( 'saved_settings.json' ):
            return

        with open('saved_settings.json' ) as data_file:
            save_data = json.load( data_file )

        self.rt.decorators.clear()
        self.decorator_types.clear()

        for elt in save_data:
            self.add_decorator( elt[0] )
            self.rt.decorators[-1].import_settings( elt[1] )

        self.redraw_decorators()

    def make_ready_pushed( self ):
        self.ready_button.config(state='disabled')
        self.t_ready = threading.Thread( target=self.make_ready_async )
        self.t_ready.daemon = True
        self.t_ready.start()

    def make_ready_async( self ):
        self.rt.make_ready()
        self.start_button.config(state='normal')

    def start_pushed( self ):
        self.start_button.config(state='disabled')
        self.t_decorate = threading.Thread( target=self.start_async )
        self.t_decorate.daemon = True
        self.t_decorate.start()
        self.screenshot_button.config(state='normal')
        self.stop_button.config(state='normal')

    def start_async( self ):
        self.rt.go_decorate()
        self.start_button.config(state='normal')
        self.screenshot_button.config(state='disabled')
        self.stop_button.config(state='disabled')

    def screenshot_pushed( self ):
        self.rt.take_screenshot()

    def stop_pushed( self ):
        self.rt.running = False

    def refresh_hwnds( self ):
        print( 'refresh triggered' )
        self.known_hwnds = self.rt.sc.get_hwnds() # [ [ hwnd, description ] ]
        self.window_list.delete( 0, tk.END )
        for i in range(len(self.known_hwnds)):
            self.window_list.insert( tk.END, self.known_hwnds[i][1] )
            if self.known_hwnds[i][0] in self.rt.hwnds:
                self.window_list.selection_set( i )

        for hwnd in self.rt.hwnds:
            if hwnd not in (x[0] for x in self.known_hwnds):
                self.rt.hwnds.remove( hwnd )

    def change_hwnds_selection( self, evt ):
        print( 'change triggered' )
        chosen = self.window_list.curselection()
        self.rt.hwnds.clear()
        for i in chosen:
            self.rt.hwnds.append( self.known_hwnds[i][0] )

    def on_close( self ):
        self.rt.detector_async.shutdown()
        self.rt.running = False
        sys.exit()

