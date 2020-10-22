import os
import pyocr
import pyocr.builders
from PIL import Image
import pdf2image
import tempfile as tmp
import tempfile2 as tmp2
#%%
#list of possible bills
list_bills = ['Eversource', 'NATIONALGRID', 'AMAZON', 'EVERSOURCE']
bills_found = list()
#%%
#create a temporary folder AND file object in which the JPEGs will be saved
temp_dir = tmp.gettempdir()
td = tmp.TemporaryDirectory(suffix=None, prefix=None, dir=temp_dir)
#td is an object, td.name shows the path and is then converted
td_new = td.name.replace(os.sep, '/')
temp_f = ''.join(('', td_new,''))
print("In case you are interested, your temporary folder is", temp_f)
#creates a temporary file object
temp_pic = tmp.TemporaryFile(mode='w+b', dir=temp_f)
#%%
#input link (in APP)
#filename of your bill in PDF
#directory where your PDF is stored
print("please specify file directory; with >\< and without letter strings please")
link = input()
#convert backslashes to slashes
new_link = link.replace(os.sep, '/')
#remove those pesky letter strings
new_link_2 = new_link.replace('"', '')
file_in = ''.join(('', new_link_2,''))
#%%
#conversion to colored image
# SPECIFY INPUT/PDF FOLDER + OUTPUT FOLDER + NO OF PAGES TO BE SCANNED; >>>NOT ZERO-INDEXED<<<
pdf2image.convert_from_path(file_in, dpi = 400, output_folder = temp_f,
                            first_page = 0, last_page = 5, fmt = 'png',
                            thread_count = 1, userpw = None, use_cropbox = False, strict = False,
                            transparent = False, output_file = temp_pic)
##'4a90245d-9f16-413f-8c8d-23469f4775db'
#%%
####GET A VERSION WITH HASHLIB WORKING!
#OCR process
tools = pyocr.get_available_tools()[0]
text = tools.image_to_string(Image.open(temp_pic),
					builder=pyocr.builders.DigitBuilder())

strings = [text]
print(text)
#%%
#search for string in text (Eversource; Nationalgrid etc.)
for names in list_bills:

    if strings.str.contains(list_bills):
        bills_found.append(names)
        #menu for confirmation
#IF FAILED
    else:
        pass
#%%
#conversion colored image to monochrome mode
#filename of your JPEG/PNG
link = temp_f.name
#convert backslashes to slashes
#new_link = link.replace(os.sep, '/')
#remove those pesky letter strings
#new_link_2 = new_link.replace('"', '')
#input_pic = ''.join(('', new_link,''))

#opens the JPEG/PNG and converts it to monochrome
img = Image.open(link)
img = img.convert('L')
img.save('monochrome_image.jpg')

#OCR process
tools = pyocr.get_available_tools()[0]
text_mono = tools.image_to_string(Image.open('monochrome_image.jpg'),
					builder=pyocr.builders.DigitBuilder())

#%%
#search for string
for n in list_bills:
	text_mono.str.contains(list_bills)

#notification: "is this bill X?"

#IF FAILED

#ask for manual input
if len(bills_found == 0):
    import Python_input_bill_list as man_input
