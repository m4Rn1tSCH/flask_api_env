import pyocr
import pyocr.builders
import pandas as pd
from PIL import Image
import os
#LEAVE r INF FRONT
#filename of your PDF/directory where your PNG is stored;regularly indexed
link = r"C:\Users\bill-\Desktop\4a90245d-9f16-413f-8c8d-23469f4775db-1.png"

new_link = link.replace(os.sep, '/')
input_directory = ''.join(('', new_link,''))

tools = pyocr.get_available_tools()[0]

text = tools.image_to_string(Image.open(input_directory),
                            builder=pyocr.builders.DigitBuilder())

#print(text)

#INACTIVE
#OCR_list =[]
#OCR_list_full = OCR_list.append(text)
#OCR_DF = pd.DataFrame(OCR_list_full)
#OCR_DF.to_csv("C:/Users/bill-/Desktop/OCR_text.csv")

