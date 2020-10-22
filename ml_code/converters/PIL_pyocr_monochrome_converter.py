##CONVERT PICTURES TO MONOCHROMATIC MODE TO ENHANCE SCAN AND REDUCE VISUAL NOISE
from PIL import Image
import os
##LEAVE r IN FRONT
#filename of your PDF/directory where your PDF is stored
link = r'C:\Users/bill-\Desktop\2016fy detailed Income Statement.pdf' 
new_link = link.replace(os.sep, '/')
input_directory = ''.join(('', new_link,''))

img = Image.open('image.jpg')
img = img.convert('L')
img.save('image.jpg')

