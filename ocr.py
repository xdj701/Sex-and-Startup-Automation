import numpy as np
import cv2
import pytesseract
import re
from pdf2image import convert_from_path
from nltk.corpus import words

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
    pages = [pytesseract.image_to_string(erode_and_dilate_image(page), lang="eng", config="--psm 3") for page in pdf]
    for i in range(len(pages)):
        page = pages[i]
        lines = page.strip("\n").split("\n\n")
        # check if the last line contains page number, file path, or doc name
        if not any([token in words.words() for token in lines[-1].split(" ")]):
            lines.pop()
            pages[i] = "\n\n".join(lines)

    text = "\n".join(pages)
    
    # Gj) -> (j)
    text = re.sub("G\w\)", lambda o: "(" + o.group()[1:], text)
    # {a) -> (a)
    text = re.sub("\{\w\)", lambda o: "(" + o.group()[1:], text)
    # 1, -> 1., 3,3.1 -> 3.3.1 This may be too hacky
    text = re.sub("(\n+\d+),(\s|\d+\.)", lambda o: ".".join(o.groups()), text)

    # WHEREOPF, WHEREOEF -> WHEREOF
    text = re.sub("WHEREO[PE]F", "WHEREOF", text)

    # non-standard quotation marks
    text = re.sub("[“”‘’]","\"", text)

    # ARTICLE OI, VOI, VHUI, VHI, VU, VIM -> ARTICLE VIII
    text = re.sub("(ARTICLE [IVX]?)(OI|HU?I|U|IM)", lambda o: o.groups()[0] + "III", text)
    # Il -> II, lI -> II, IlI -> III
    text = re.sub("(ARTICLE [IVX]*)(l+)([IVX]*)", lambda o: o.groups()[0] + "I"*len(o.groups()[1]) + o.groups()[2], text)
    # IT -> II, IE -> II, XE -> XI
    text = re.sub("(ARTICLE [IVX]*)[TE]", lambda o: o.groups()[0] + "I", text)
    # ii -> II, il -> II, ili -> IIi
    text = re.sub("(ARTICLE [IVX]*)i[il]", lambda o: o.groups()[0] + "II", text)
    # ii -> II, iI -> II, IIi -> III
    text = re.sub("(ARTICLE [IVX]*)(i+)", lambda o: o.groups()[0] + "I"*len(o.groups()[1]), text)
    # iX -> IX, 1X -> IX
    text = re.sub("ARTICLE [1i]X", "ARTICLE IX", text)

    return text.replace("CERTIFICATE ON ORPORATION", "CERTIFICATE OF CORPORATION"
        ).replace("CERTIFICATE  ORPORATION", "CERTIFICATE OF CORPORATION"
        ).replace("ARTICLE |", "ARTICLE I"
        ).replace("ARTICLE It", "ARTICLE III"
        ).replace("ARTICLE If", "ARTICLE II"
        ).replace("ARTICLE hit", "ARTICLE III"
        ).replace("ARTICLE [V", "ARTICLE IV"
        ).replace("ARTICLE VH", "ARTICLE VII"
        ).replace("(ili)", "(iii)"
        ).replace("shal]", "shall"
        ).replace("&)", "(x)")
