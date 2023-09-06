#Transform from jpg to docx

from matplotlib.pyplot import text
from PIL import Image
from docx import Document

img = Image.open("dilekçe.jpg") ## or jpg and apple applys

doc = Document()
doc.add_paragraph(text)
doc.save("None.docx") ## or pdf if user what do wants