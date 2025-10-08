import PyPDF2
from PIL import Image
import os

def extract_images_from_pdf(pdf_path, output_folder):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            resources = page['/Resources']
            if '/XObject' in resources:
                xObject = resources['/XObject'].get_object()
                
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].get_data()
                        if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                            mode = "RGB"
                        else:
                            mode = "P"
                        if '/Filter' in xObject[obj]:
                            if xObject[obj]['/Filter'] == '/FlateDecode':
                                img = Image.frombytes(mode, size, data)
                                img.save(os.path.join(output_folder, f'{obj[1:]}.png'))
                            elif xObject[obj]['/Filter'] == '/DCTDecode':
                                img = open(os.path.join(output_folder, f'{obj[4:]}.jpg'), "wb")
                                img.write(data)
                                img.close()
                            elif xObject[obj]['/Filter'] == '/JPXDecode':
                                img = open(os.path.join(output_folder, f'{obj[1:]}.jp2'), "wb")
                                img.write(data)
                                img.close()
                            elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                                img = open(os.path.join(output_folder, f'{obj[1:]}.tiff'), "wb")
                                img.write(data)
                                img.close()
                        else:
                            img = Image.frombytes(mode, size, data)
                            img.save(os.path.join(output_folder, f'{obj[1:]}.png'))
            else:
                print(f"No '/XObject' found in page {page_num + 1} resources.")

# Usage
pdf_path = 'a.pdf'
output_folder = 'output_images'
extract_images_from_pdf(pdf_path, output_folder)
