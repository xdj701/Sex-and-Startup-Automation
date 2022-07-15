import math
import os
import numpy as np
import cv2
import pytesseract
import pdfplumber
import re

from typing import Tuple, Union
from PIL.PpmImagePlugin import PpmImageFile
from pdf2image import convert_from_path
from deskew import determine_skew
from nltk.corpus import words

MEANINGFUL_WORDS = [word.lower() for word in words.words()]

# Rotate an image by an angle
def rotate_matrix(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:

    if angle is None:
        return image
    #else:
    #    print("Rotate angle: {}".format(angle))
    
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# Identify underlines and draw countours to cover them
def remove_underlines(image: np.ndarray) -> np.ndarray:
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # construct a horizontal kernel (30 x 1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # An in-place operation. use a 2-pt thick white line to cover the underlines.
        cv2.drawContours(image, [c], -1, (255,255,255), 2)
    return image

# Apply computer vision techniques to improve the quality of an image
def preprocess_image(img: PpmImageFile) -> np.ndarray:
    # Convert a PIL image to the numpy format for OpenCV
    np_img = np.array(img)

    # Convert RGB to BGR 
    bgr_img = np_img[:, :, ::-1].copy()
    
    # Convert image to grayscale
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Rotate the image by a detected skewed angle.
    angle = determine_skew(gray_img, num_peaks=10)
    if abs(angle) < 5.0:
        rotated_img = rotate_matrix(gray_img, angle, (255, 255, 255))
    else:
        print("the skewed angle {} is too big!".format(angle))
        rotated_img = gray_img
        
    # remove lines under words
    cleaned_img = remove_underlines(rotated_img)
    
    # Apply a square Gaussian kernel (3x3) to smooth the image
    smoothed_img = cv2.GaussianBlur(cleaned_img, (3, 3), 0)

    # Use a square kernel (2x2) to erode a character to make its edges sharper.
    eroded_img = cv2.erode(smoothed_img, np.ones((2,2), np.uint8), iterations=1)

    # Use a rectangular kernel (2x1) to dilate a character to make its edges bolder
    dilated_img = cv2.dilate(eroded_img, np.ones((2,1), np.uint8), iterations=1)

    return dilated_img

# If a line is made up of meaningless tokens
def is_meaningless_line(line: str) -> bool:

    # document name
    if any(x in line for x in [".doc", ".pdf", "DocuSign Envelope ID", 
                               "(TUE)", "FROM CORPORATION TRUST", "CORPORATE TRUST CENTER"]):
        return True
    
    tokenIsMeaningless = []
    for token in line.lower().replace(".", " ").split(" "):
        # page number
        if re.match("^\d+$", token) is not None:
            tokenIsMeaningless.append(True)
        # gibberish (e.g., GDS VF&H\1642487.6 5)
        elif token not in MEANINGFUL_WORDS:
            tokenIsMeaningless.append(True)
        else:
            tokenIsMeaningless.append(False)

    return all(tokenIsMeaningless)

# Clean the margins of a page
def remove_header_and_footer(lines: [str]) -> str:
    
    clean_lines = lines
    if is_meaningless_line(clean_lines[0]):
        clean_lines = clean_lines[1:]
        # one more line
        if is_meaningless_line(clean_lines[0]):
            clean_lines = clean_lines[1:]
    if is_meaningless_line(clean_lines[-1]):
        clean_lines = clean_lines[:-1] 
        if is_meaningless_line(clean_lines[-1]):
            clean_lines = clean_lines[:-1]

    return "\n\n".join(clean_lines)

def fix_pdf_errors(raw_text: str) -> str:
    # ( a) -> (a), ( 1) -> (1)
    clean_text = re.sub("\( +(.*)", lambda o: "(" + o.groups()[0], raw_text)
    # (a ) -> (a)
    clean_text = re.sub("(.*) +\)", lambda o: o.groups()[0] + ")", clean_text)
    # (o r) -> (or), (o ther) -> (other)
    clean_text = re.sub("\(o (\w)", lambda o: "(o" + o.groups()[0], clean_text)
    # SeriesC -> Series, Section1 -> Section 1, Article I -> Article I
    processed_text = re.sub("(eries|ection|rticle|RTICLE)([0-9A-Z])", lambda o: " ".join(o.groups()), processed_text)
    
    return clean_text.replace("P referred", "Preferred"
        ).replace("oft he", "of the")

def correct_mispelled_symbols(raw_text: str) -> str:
    
    processed_text = raw_text
    
    MISPELLING_DICT = {
        "|": "", 
        "—": " ", 
        "_": " ", 
        "- ": " ", 
        "{": "(", 
        "}": ")", 
        "\(": ")(", 
        "©": "c", 
        "(¢)": "(e)", 
        "(ce)": "(e)", 
        "(ec)": "(e)", 
        "(hb)": "(h)", 
        "(ji)": "(ii)", 
        "(iti)": "(iii)", 
        "(vy)": "(v)", 
        "(vili)": "(viii)", 
        "[I": "I", 
        "TV(": "IV(", 
        "Cc.": "C.", 
        "Dz.": "D.", 
        "l0": "10", 
        "-Iwo": "-Two", 
        "FIr tH": "FIFTH", 
        "TWELI1IH" : "TWELFTH", 
        "Article 5S": "Article 5", 
        "Series AS": "Series A5", 
        "Series A5S": "Series A5", 
        "Series A6é": "Series A6", 
        "AT Preferred": "A7 Preferred", 
        "Series A110": "Series A10",
        "1, THE UNDERSIGNED": "I, THE UNDERSIGNED",
    }
    
    for mispelled_symbol, correct_symbol in MISPELLING_DICT.items():
        processed_text = processed_text.replace(mispelled_symbol, correct_symbol)
    # @
    processed_text = re.sub("@\)?", "(i)", processed_text)
    # qd), qi) ,ql) -> (1)
    processed_text = re.sub("q[dil]\)", "(1)", processed_text)
    # qd -> (d)
    processed_text = re.sub("\nqd ", "(d) ", processed_text)
    # Gj) -> (j)
    processed_text = re.sub("G\w{1,4}\)", lambda o: "(" + o.group()[1:], processed_text)
    # [X -> IX, [V -> IV, 1V -> IV, I'V -> IV
    processed_text = re.sub("(\[|1|I')(X|V)", lambda o: "I" + o.groups()[1], processed_text)
    # \n(0) -> (o)
    processed_text = re.sub("\n\(0\) ", "\n(o) ", processed_text)
    # (iXE) -> (i)(E)
    processed_text = re.sub("(\(\w+)X(\w+\))", lambda o: ")(".join(o.groups()), processed_text)
    # (iMA) -> (i)(A)
    processed_text = re.sub("(\(\w+)M(\w+\))", lambda o: ")(".join(o.groups()), processed_text)
    # )iiX( -> )(ii)(
    processed_text = re.sub("\)(i+)X\(", lambda o: ")(" + len(o.groups()[0])*"i" + ")(", processed_text)
    # Il -> II, Ill -> III
    processed_text = re.sub("I(l+)", lambda o: "I"*len(o.group()), processed_text)
    # -l, -I, -],-! -> -1
    processed_text = re.sub("-[lI\]\!]", "-1", processed_text)
    # L. -> 1.
    processed_text = re.sub("(\s)L\.", lambda o: o.groups()[0] + "1.", processed_text)  
    # -? -> -2
    processed_text = re.sub("-[\?]", "-2", processed_text)
    # 13" day -> 13th day
    processed_text = re.sub("(\d)\" day", lambda o: o.group()[0] + "th day", processed_text)
    # Series A11
    processed_text = re.sub("Series Al[1l\]]", "Series A11", processed_text)
    
    return processed_text

def correct_mispelled_patterns(raw_text: str) -> str:
    processed_text = raw_text
    # a! -> al
    processed_text = re.sub("a!", "al", processed_text) 
    # all (l), l], ll], l!, lj)
    processed_text = re.sub("(a|e|i|A)l{1,2}[\)\]\!j]", lambda o: o.groups()[0] + "ll", processed_text)   
    # adjustment (adjusment, adiustment, adjusi:ment, adiusiiument) NOTE: advancement
    processed_text = re.sub("(a|A)d[ijmnrstu1:]{4,}ent", lambda o: o.groups()[0] + "djustment", processed_text)
    # affirmative (affimative, affiiiative, affiisuative)
    processed_text = re.sub("af[a-z1:]{3,5}ative", "affirmative", processed_text)
    # article
    processed_text = re.sub("AR[UL]ICLE", "ARTICLE", processed_text)    
    # required, acquired
    processed_text = re.sub("auired", "quired", processed_text)
    # certificate
    processed_text = re.sub("CERTIFICA1[r\&]", "CERTIFICATE", processed_text)
    # Convert (Conveii, Convesi)
    processed_text = re.sub("(c|C)on[vy]e[is]i(ed|ible)", lambda o: o.groups()[0] + "onvert" + o.groups()[1], processed_text)
    # determin (deteimed, deterined, deteiiiiined, deteriiination, delauination)
    processed_text = re.sub("(d|D)ete[a-z1:\s]{2,4}in(a|e|i)", lambda o: o.groups()[0] + "etermin" + o.groups()[1], processed_text)
    processed_text = re.sub("(d|D)elauin(a|e|i)", lambda o: o.groups()[0] + "etermin" + o.groups()[1], processed_text)
    # distribute (disizibute, disiribute)
    processed_text = re.sub("disi[rz]ib", "distrib", processed_text)    
    # each (cach)
    processed_text = re.sub("(\W)cach(\W)", lambda o: o.groups()[0] + "each" + o.groups()[1], processed_text)
    # file
    processed_text = re.sub("FIT[EFR\.]", "FILE", processed_text)
    # firm (fii, fiun, fiiii, fiiin)
    processed_text = re.sub("(\W)fi[iu]+n?(\W)", lambda o: o.groups()[0] + "firm" + o.groups()[1], processed_text)
    # forth (finin, foxin)
    processed_text = re.sub("(\W)f(ini|oxi)n(\W)", lambda o: o.groups()[0] + "forth" + o.groups()[2], processed_text)
    # form (fox, fou, fori, fous, foiin) NOTE: for
    processed_text = re.sub("(\Win)( the | )fo[a-z]{1,3}", lambda o: o.groups()[0] + o.groups()[1] + "form", processed_text)
    # formed, former, forming (foisu, foiin, foun, foim, foiim)
    processed_text = re.sub("fo[imnrsu:1\.]{2,3}(ed|er|in)", lambda o: "form" + o.groups()[0], processed_text)
    # formula (fouuula)
    processed_text = re.sub("(f|F)o[imnrsu:1\.]{2}ula", lambda o: o.groups()[0] + "ormula", processed_text)
    # from (fium)
    processed_text = re.sub("(\W)f(iu)m(\W)", lambda o: o.groups()[0] + "from" + o.groups()[2], processed_text)
    # general
    processed_text = re.sub("(g|G)en[ec]ral", lambda o: o.groups()[0] + "eneral", processed_text)  
    # harm (harin, haiiu)
    processed_text = re.sub("ha[riun]{3}(\W|s|ful|less)", lambda o: "harm" + o.groups()[0], processed_text)
    # holders (holdes, holdas, holdcrs)
    processed_text = re.sub("hold(a|e|cr)s", "holders", processed_text)
    processed_text = re.sub("each bolder", "each holder", processed_text)
    # In
    processed_text = re.sub("[1jJ]n", "In", processed_text)
    # indemnif (indeiunif, Indeminif)
    processed_text = re.sub("(i|I)nde[imu]{2}nif", lambda o: o.groups()[0] + "ndemnif", processed_text)
    # judgment (judgiuent)
    processed_text = re.sub("judg[a-z1:]{2,4}t", "judgment", processed_text)
    # net (nct, nct
    processed_text = re.sub("(\W)nct(\W)", lambda o: o.groups()[0] + "net" + o.groups()[1], processed_text)
    # of
    processed_text = re.sub("(\W)(o:|0;|af)(\W)", lambda o: o.groups()[0] + "of" + o.groups()[2], processed_text)
    # or
    processed_text = re.sub("(\W)(ar|oF)(\W)", lambda o: o.groups()[0] + "or" + o.groups()[2], processed_text)
    # partner (pasiner, pasiuer, pariner, pariuer)
    processed_text = re.sub("pa[inrsu]{3}er", "partner", processed_text)
    # per (por)
    processed_text = re.sub("(\W)p(o)r(\W)", lambda o: o.groups()[0] +"per" + o.groups()[2], processed_text)
    # permit (parmit, peiimit, pesmit, perinit, periuit, painit, pe:mit)
    processed_text = re.sub("(\W)p[a-z1:]{3,4}it(\W|s|t)", lambda o: o.groups()[0] + "permit" + o.groups()[1], processed_text)
    # referred (refced, refccd, reféued, reftxaed; iefeaved, isiemed)
    processed_text = re.sub("ref[a-z1:é]{1,3}[ce]d(\W)", lambda o: "referred" + o.groups()[0], processed_text)
    processed_text = re.sub("i(efeav|siem)ed", "referred", processed_text)
    # regist (registi, regisir)
    processed_text = re.sub("regist?ir?", "registr", processed_text)
    # ring (rmg)
    processed_text = re.sub("rmg", "ring", processed_text)
    # Series (Scries, serics)
    processed_text = re.sub("(s|S)(crie|eric)s", lambda o: o.groups()[0] + "eries", processed_text)    
    # secretary (secre, secre', secreta, secreta:)
    processed_text = re.sub("(ECRE|ecre)(ta)?[':]? ", lambda o: "ECRETARY " if o.groups()[0][0].isupper() else "ecretary ", processed_text)
    # set
    processed_text = re.sub("(\W)s[co]t(\W)", lambda o: o.groups()[0] + "set" + o.groups()[1], processed_text)
    # shall (shail, sholl)
    processed_text = re.sub("sh(ai|ol)l", "shall",processed_text)       
    # te1mina, teiaimina, teialmina, teiiina
    processed_text = re.sub("(t|T)e[a-z1:]{2,4}ina", lambda o: o.groups()[0] + "ermina", processed_text)
    # terms (ters, tems, teims, teriis, te:ms, te::us, teriiis; teitas)
    processed_text = re.sub("(\W[tT])e[irmnustz1:]{1,4}s(\W)", lambda o: o.groups()[0] + "erms" + o.groups()[1], processed_text)
    processed_text = re.sub("(\W)tei[int]{1,2}(\W)", lambda o: o.groups()[0] + "term" + o.groups()[1], processed_text)
    processed_text = re.sub("t(azu8|eitas)", "terms", processed_text) 
    # the (thc)
    processed_text = re.sub("(\W)thc(\W)", lambda o: o.groups()[0] + "the" + o.groups()[1], processed_text)
    # to (fo)
    processed_text = re.sub("(\W)fo(\W)", lambda o: o.groups()[0] + "to" + o.groups()[1], processed_text)
    # WHEREOF
    processed_text = re.sub("WHEREO[EFP]{1,2}F", "WHEREOF", processed_text)
    
    return processed_text

def correct_mispelled_words(raw_text: str) -> str:

    MISPELLING_DICT = {
        # phrases
        "as 2 result": "as a result",       
        "as neatly as": "as nearly as",
        "by their teams": "by their terms",
        "Inthe": "In the",
        "Jess than": "less than",
        "tothe": "to the",
        " ofa ": " of a ",
        # words
        "snd": "and",
        "aacange": "arrange",
        "aducas": "address",
        "adveise": "adverse",
        "aiie": "ame",
        "agrecin": "agreem",
        "anend": "amend",
        "Anthor": "Author",
        "aveiage": "average",
        "ay:ce": "agree",
        "attomey": "attorney",
        "Boud": "Board",
        "Compeny": "Company",
        "canital": "capital",
        "cci": "cei",
        "confeed": "conferred",
        "conyer": "conver",
        "concure": "concurre",
        "Corpoiat": "Corporat",
        "cqual": "equal",
        "cicise": "ercise",
        "crcase": "crease",
        "decm": "deem",
        "dexign": "design",
        "dixect": "direct",
        "ccd": "eed",
        "efiect": "effect",
        "cither": "either",
        "cmploy": "employ",
        "ercice": "ercise",
        "fallest": "fullest",
        "fimds": "funds",
        "gencics": "gencies",
        "govem": "govern",
        "gxant": "grant",
        "yxoup": "group",
        "heicin": "herein",
        "HEAT.TH": "HEALTH",
        "i.c.": "i.e.",
        "imsofar": "insofar",
        "invesi": "invest",
        "intermst": "interest",
        "indebtermss": "indebtness",
        "[ss": "Iss",
        "\\ater": "later",
        "Taw": "Law",
        "1aw": "law",
        "1evel": "level",
        "licu": "lieu",
        "Iess": "Less",
        "mect": "meet",
        "NAMB" : "NAME",
        "nocd": "need",
        "occui": "occurr",
        "eccumci": "occurren",
        "onverm": "onverti",
        "orgumiz": "organiz",
        "othciwisc": "otherwise",
        "ounnon": "ommon",
        "paia": "para",
        "pesson": "person",
        "poweis": "powers",
        "prefsences": "preferences",
        "zaice": "rance",
        "REGIS1": "REGIST",
        "Tepre": "repre",
        "respeot": "respect",
        "resirict": "restrict",
        "aaid": "said",
        "securitics": "securities",
        "SECIION": "Section",
        "Fisiewed": "Series",
        "sirat": "strat",
        "stract": "street",
        "Stook": "Stock",
        "sr1OCK": "STOCK",
        "snant": "suant",
        "vecial": "pecial",
        "Tefer": "refer",
        "Tights": "rights",
        "uior": "rior",
        "wansfer": "transfer",
        "iansact": "transact",
        "ireat": "treat",
        "warunt": "warrant",
        "WIItNESS": "WITNESS",
        "yccs": "yees",
    }
    
    processed_text = raw_text
    for mispelled_word, correct_word in MISPELLING_DICT.items():
        processed_text = processed_text.replace(mispelled_word, correct_word)

    return processed_text

def clean_format(raw_text: str) -> str:
    
    processed_text = re.sub(" +", " ", raw_text)
    
    # ii) -> (ii), J) -> (J)
    # note: this may cause other issue are) -> (are)
    processed_text = re.sub("\n([ivx0-9A-Z]{,3}\))", lambda o: "\n(" + o.groups()[0], processed_text)
    # ( a) -> (a)
    processed_text = re.sub("\( (.*)", lambda o: "(" + o.groups()[0], processed_text)
    # (a ) -> (a)
    processed_text = re.sub("(.*) \)", lambda o: o.groups()[0] + ")", processed_text)
    # (i)p -> (i) p
    processed_text = re.sub("\)(\w)", lambda o: ") " + o.groups()[0], processed_text)
    # (c(i -> (c)(i)
    processed_text = re.sub("\((\w+)\(", lambda o: "(" + o.groups()[0] + ")(", processed_text)
    # ) i) ->  )(i)
    processed_text = re.sub("\) (i+)\)", lambda o: ")(" + len(o.groups()[0])*"i" + ")", processed_text)
    # ) ‘I -> ) I
    processed_text = re.sub("\) ‘(\w)", lambda o: ") " + o.groups()[0], processed_text)
    # (i)) -> (i)
    processed_text = re.sub("(\n\(\w\))\) ", lambda o: o.groups()[0] + " ", processed_text)
    
    # 1, The name -> 1. (only for New Paragraph)
    processed_text = re.sub("(\n\n\d), ([A-Z])", lambda o: ". ".join(o.groups()), processed_text)
    # 2 Dividends -> 2. Dividends (only for New Paragraph)
    processed_text = re.sub("(\n\n\d) ([A-Z])", lambda o: ". ".join(o.groups()), processed_text)
    # SeriesC -> Series, Section1 -> Section 1, Article I -> Article I
    processed_text = re.sub("(eries|ection|rticle|RTICLE)([0-9A-Z])", lambda o: " ".join(o.groups()), processed_text)
    
    return processed_text.replace(" .", "."
        ).replace("..", "."
        ).replace("((", "("
     )
    
def fix_ocr_errors(raw_text: str) -> str:
    
    processed_text = raw_text
    
    processed_text = correct_mispelled_symbols(processed_text)

    processed_text = correct_mispelled_patterns(processed_text)
 
    processed_text = correct_mispelled_words(processed_text)   

    processed_text = clean_format(processed_text)
    
    return processed_text

def run_ocr_correction_test(testdir_path: str):
    for file_name in os.listdir(testdir_path):
        if "OCR.txt" in file_name:
            print("=={}==".format(file_name))
            identical = True

            ocr_file_path = testdir_path + "/" + file_name   
            baseline_file_path = testdir_path + "/" + file_name.replace("OCR", "Baseline")        
            ocr_file = open(ocr_file_path, 'r', encoding="utf8")
            ocr_content = "".join(ocr_file.readlines())
            baseline_file = open(baseline_file_path, 'r', encoding="utf8")
            baseline_content = "".join(baseline_file.readlines())
            for fix_ocr_line, baseline_line in zip(fix_ocr_errors(ocr_content).split("\n"), baseline_content.split("\n")):
                if fix_ocr_line != baseline_line:
                    print("fixed ocr: " + fix_ocr_line)
                    print("baseline: " + baseline_line)
                    print("")
                    identical = False
            ocr_file.close()
            baseline_file.close()
            print("PASSED" if identical else "FAILED")
            
def scan_pdf(file_path: str) -> str:

    with pdfplumber.open(file_path) as pdf:
        page_texts = [page.extract_text(layout=True).strip() for page in pdf.pages]
        # the pdf is machine-generated
        if all(page_texts):
            print("a machine-generated file")
            clean_texts = []
            for page_text in page_texts:
                raw_page_text = re.sub("[\n]{2,}", "\n\n", page_text)
                lines = [re.sub("[\s\n]+", " ", line).strip() for line in raw_page_text.split("\n\n")]
                clean_texts.append(try_remove_header_and_footer(lines))
            full_text = "\n\n".join(clean_texts)
            return fix_pdf_errors(full_text)
        # the pdf is a scanned copy, try OCR
        else:
            print("an OCR-scanned file")
            page_imgs = convert_from_path(file_path)

            # return a list of text in each page of the input doc
            page_texts = [pytesseract.image_to_string(preprocess_image(page), lang="eng", config="--psm 3") for page in page_imgs]
            raw_texts = []
            for page_text in page_texts:
                lines = page_text.strip("\n").split("\n\n")
                raw_texts.append(try_remove_header_and_footer(lines))   
            full_text = "\n\n".join(raw_texts)
            return fix_ocr_errors(full_text)
