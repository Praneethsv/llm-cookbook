import pdfplumber

from PIL import Image

import io


images = []
with pdfplumber.open('/home/sv/Downloads/Nexus-Downloads/BackUp-Downloads/Venkata_Praneeth_CV_Updated.pdf') as pdf:
   for page in pdf.pages:
     text = page.extract_text()
     tables = page.extract_tables()
     images = page.images



# for el in image:
#    print(el['stream'])

# exit(1)

image = images[0]

pdf_stream = image['stream']

image_bytes = pdf_stream.get_data()

stream = io.BytesIO(image_bytes)

pil_image = Image.open(stream)

pil_image.show()

