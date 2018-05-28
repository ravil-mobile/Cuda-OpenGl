# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from PIL import Image
import numpy
import argparse
import os

parser = argparse.ArgumentParser(description="Generate flag file from image.")
parser.add_argument("image_file", type = str)
args = parser.parse_args()
image_file = args.image_file
#find absolute path if relative is provided
image_file = os.path.abspath(image_file)

im = Image.open(image_file).convert("L")
#im = Image.open("./mesh.jpg").convert("L") 
im = im.rotate(180)
im = im.transpose(Image.FLIP_LEFT_RIGHT)
image = numpy.array( im )
row_size, col_size = image.shape

fluid = '0'
wall = '1'
mov_wall = '2'
inflow = '3'
outflow = '4'
separator = ' '

with open ("mesh.txt", "w") as outfile: 
    for row in range(row_size):
        if (row == 0 or row == (row_size - 1 ) ):
            outfile.write( separator.join(col_size * [wall] ) + '\n' )
        else:
            output_string = [ str(inflow) ]
            # don't consider boundary pixel values
            output_string.extend ( map( lambda x: fluid if (x >= 200) else wall, 
                                   image[row, 1 : -1 ]) )
            output_string.extend([outflow, '\n'])
            output_string = separator.join(output_string)
            outfile.write(output_string)        