import numpy as np
import matplotlib.pyplot as plt
import io
import base64    
    
HEADER = '<!doctype html><html lang="ja"><body>'
TEMPLATE = '<img src="data:image/png;base64,{image_bin}">'
FOOTER = '</body></html>'    
    
class HtmlWriter:
    
    def __init__(self):
        self.html = HEADER    
    
    def add_fig(self, fig):     
        sio = io.BytesIO()
        fig.savefig(sio, format='png')
        image_bin = base64.b64encode(sio.getvalue())
        image_html = TEMPLATE.format(image_bin=str(image_bin)[2:-1])
        self.html += image_html 
        
    def write(self, filepath):
        with open(filepath, "w") as f:
            f.write(self.html + FOOTER)
        
    
    
    