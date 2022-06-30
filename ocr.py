import numpy as np
import cv2
import pytesseract
import re
from pdf2image import convert_from_path

def erode_and_dilate_image(img):
    np_img = np.array(img) 
    # Convert RGB to BGR 
    img = np_img[:, :, ::-1].copy()
    
    # Use a smaller kernel to look at an adjacent character
    kernel = np.ones((2,1), np.uint8)

    # A must step for processing bolded characters
    eroded_img = cv2.erode(img, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    return dilated_img

# return a string
def scan_pdf(pdf_path):
    pdf = convert_from_path(pdf_path)

    # return a list of text in each page of the input doc
    raw_texts = [pytesseract.image_to_string(erode_and_dilate_image(page), lang="eng", config="--psm 3") for page in pdf]

    # TODO: try to remove the page number, doc name, or file path at the bottom of each page
    text = "\n".join(raw_texts)
    
    # Gj) -> (j)
    text = re.sub("G\w\)", lambda o: "(" + o.group()[1:], text)
    # {a) -> (a)
    text = re.sub("\{\w\)", lambda o: "(" + o.group()[1:], text)
    # 1, -> 1., 3,3.1 -> 3.3.1 This may be too hacky
    text = re.sub("(\n+\d+),(\s|\d+\.)", lambda o: ".".join(o.groups()), text)

    # ARTICLE OI, VOI, VHUI, VU, VIM -> ARTICLE VIII
    text = re.sub("(ARTICLE [IVX]?)(OI|HUI|U|IM)", lambda o: o.groups()[0] + "III", text)
    # Il -> II, lI -> II, IlI -> III
    text = re.sub("(ARTICLE [IVX]*)(l+)([IVX]*)", lambda o: o.groups()[0] + "I"*len(o.groups()[1]) + o.groups()[2], text)

    return text.replace("CERTIFICATE ON ORPORATION", "CERTIFICATE OF CORPORATION"
        ).replace("CERTIFICATE  ORPORATION", "CERTIFICATE OF CORPORATION"
        ).replace("WHEREOPF", "WHEREOF"
        ).replace("WHEREOEF", "WHEREOF"
        ).replace("ARTICLE It", "ARTICLE III"
        ).replace("ARTICLE IIT", "ARTICLE III"
        ).replace("ARTICLE [V", "ARTICLE IV"
        ).replace("(ili)", "(iii)"
        ).replace("shal]", "shall"
        ).replace("&)", "(x)"
        ).replace("“", "\""
        ).replace("”", "\""
        ).replace("‘", "\""
        ).replace("’", "\"")
