import numpy as np
import cv2
import pytesseract
import pdfplumber
import re
from pdf2image import convert_from_path
from nltk.corpus import words

def erode_and_dilate_image(img):
    np_img = np.array(img) 
    # Convert RGB to BGR 
    bgr_img = np_img[:, :, ::-1].copy()
    #Convert image to grayscale
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # Use a smaller kernel to look at an adjacent character
    kernel = np.ones((2,1), np.uint8)

    # A must step for processing bolded characters
    eroded_img = cv2.erode(gray_img, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    return dilated_img

def fix_pdf_errors(raw_text):
    # (a ) -> (a)
    clean_text = re.sub("\( ?(\w+) ?\)", lambda o: "(" + o.groups()[0] + ")", raw_text)
    
    return clean_text.replace("P referred", "Preferred"
        ).replace("ARTICLEX", "ARTICLE X")

def fix_ocr_errors(raw_text):
    
    # Gj) -> (j)
    text = re.sub("G\w\)", lambda o: "(" + o.group()[1:], raw_text)
    # {a) -> (a)
    text = re.sub("\{\w\)", lambda o: "(" + o.group()[1:], text)
    # 1, -> 1., 3,3.1 -> 3.3.1 This may be too hacky
    text = re.sub("(\n+\d+),(\s|\d+\.)", lambda o: ".".join(o.groups()), text)
    # Cc. -> C.
    text = re.sub("^Cc\.", "C.", text)

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
        ).replace("ARTICLE V1", "ARTICLE VI"
        ).replace("ARTICLE ¥", "ARTICLE V"
        ).replace("(ili)", "(iii)"
        ).replace("shal]", "shall"
        ).replace("&)", "(x)")

def try_remove_footer(lines):
    if ".doc" in lines or all([token not in words.words() for token in lines[-1].split(" ")]):
        return "\n\n".join(lines[:-1])
    return "\n\n".join(lines)

# return a string  
def scan_pdf(file_path):

    with pdfplumber.open(file_path) as pdf:
        page_texts = [page.extract_text(layout=True).strip() for page in pdf.pages]
        # the pdf is machine-generated
        if all(page_texts):
            print("a machine-generated file")
            clean_texts = []
            for page_text in page_texts:
                raw_page_text = re.sub("[\n]{2,}", "\n\n", page_text)
                lines = [re.sub("[\s\n]+", " ", line).strip() for line in raw_page_text.split("\n\n")]
                clean_texts.append(try_remove_footer(lines))
            full_text = "\n\n".join(clean_texts)
            return fix_pdf_errors(full_text)
        # the pdf is a scanned copy, try OCR
        else:
            print("an OCR-scanned file")
            page_imgs = convert_from_path(file_path)

            # return a list of text in each page of the input doc
            page_texts = [pytesseract.image_to_string(erode_and_dilate_image(page), lang="eng", config="--psm 3") for page in page_imgs]
            raw_texts = []
            for page_text in page_texts:
                lines = page_text.strip("\n").split("\n\n")
                raw_texts.append(try_remove_footer(lines))   
            full_text = "\n\n".join(raw_texts)
            return fix_ocr_errors(full_text)


