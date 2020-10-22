import pdf2image
import os
'''
Pdf2image and PyPDF are more suitable to extract tables and entire pages
'''

#LEAVE r IN FRONT
#filename of your PDF/directory where your PDF is stored
link = r"C:\Users\bill-\Desktop\test_pic.pdf"
new_link = link.replace(os.sep, '/')
pdf_path = ''.join(('', new_link,''))


# SPECIFY INPUT/PDF FOLDER + OUTPUT FOLDER + NO OF PAGES TO BE SCANNED
pdf2image.convert_from_path(pdf_path, dpi=800, output_folder= 'C:/Users/bill-/Desktop/',
                    first_page=0, last_page=13, fmt='png',
                    thread_count=1, userpw=None, use_cropbox=False, strict=False, transparent=False,
                    output_file='4a90245d-9f16-413f-8c8d-23469f4775db')
