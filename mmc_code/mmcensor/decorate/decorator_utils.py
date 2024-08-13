import tkinter as tk
class feature_selector:

    def populate_frame( self, frame, classes, selected_classes ):
        for w in frame.winfo_children():
            w.destroy()

        self.classes = classes
        self.intvars = []
        for i in range(len(classes)):
            iv = tk.IntVar(value=classes[i] in selected_classes)
            tk.Checkbutton( frame, text=self.classes[i],onvalue=1,offvalue=0,variable=iv).grid(row=i,column=0)
            self.intvars.append(iv)

    def get_selected_classes( self ):
        out = []
        for i in range(len(self.classes)):
            if( self.intvars[i].get() ):
                out.append( self.classes[i] )

        return out 

