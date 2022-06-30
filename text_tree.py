import os, sys
import re, itertools
import numpy as np

OathPattern = r"^(IN WITNESS WHEREOF|I, UNDERSIGNED|THE UNDERSIGNED DECLARES|IT IS HEREBY DECLARED|Executed (on|at))"

ArticleArabicPattern = r"^ARTICLE \d+"
ArticleRomanPattern = r"^ARTICLE [IVXL]+"
RomanPattern = r"^[IVXL]+\.?"
OrdinalPattern = r"^([A-Z]+ST|[A-Z]+ND|[A-Z]+RD|[A-Z]+TH)[:.]"
CardinalPattern = r"^(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|[A-Z]+TEEN):\s"
AllowedTopLevelPatterns = [ArticleArabicPattern, ArticleRomanPattern, OrdinalPattern, CardinalPattern]

ParentheseArabicPattern = r"^\(\d+\)\s"
ParentheseLowercaseRomanPattern = r"^\([ivxl]+\)\s"
ParentheseUppercaseLetterPattern = r"^\([A-Z]+\)\s"
ParentheseLowercaseLetterPattern = r"^\([a-z]+\)\s"
ArabicPattern = r"^\d+\.?\s"
SubArabicPattern = r"^\d+\.\d+\.?\s"
SubSubArabicPattern = r"^\d+\.\d+\.\d+\.?\s"
UppercaseRomanPattern = r"^[IVXL]+\.?\s"
# Limit to one character to avoid matching with random string
# {1} can be replaced with + after we finish cleaning up bottom line of docs
UppercaseLetterPattern = r"^[A-Z]{1}\.?\s"
AllowedSubLevelPatterns = [ParentheseArabicPattern, ParentheseLowercaseRomanPattern, ParentheseUppercaseLetterPattern, ParentheseLowercaseLetterPattern, 
                             ArabicPattern, SubArabicPattern, SubSubArabicPattern, UppercaseRomanPattern, UppercaseLetterPattern]

class Node:
    pattern: str
    text: str
    child_nodes: []
        
    def __init__(self, pattern, full_text):
        self.pattern = pattern
        self.text = full_text
        self.child_nodes = []
    
    def adopt_children(self, new_children):
        self.child_nodes.extend(new_children)
        
    def append_text(self, new_text):
        self.text+=(" " + new_text)

    def display_family(self):
        print(self.text + "\n")
        [child.display_family() for child in self.child_nodes]
    
    def display_children(self):
        [print(child.text) for child in self.child_nodes]   
    
    def deep_clean(self):
        self.text = re.sub("\n+", " ", self.text)
        [child.deep_clean() for child in self.child_nodes]
        

def find_parent_node(parent_node, matched_level):
    if len(parent_node.child_nodes) == 0:
        return parent_node
    
    last_child = parent_node.child_nodes[-1]
    if last_child.pattern == matched_level:
        return parent_node
    else:
        return find_parent_node(last_child, matched_level)

# return a list of paragraphs
def pre_clean(raw_text):
    # remove the content between the full restatement and the oath clause
    clean_text = re.sub("\* \* \*.*IN WITNESS WHEREOF", "IN WITNESS WHEREOF", raw_text, 0, re.DOTALL)
    
    # sometimes the titles are separated by only one newline
    parsed_text = re.sub("\n+(ARTICLE \w+)\n+", lambda o: "\n\n" + o.groups()[0] + "\n\n", clean_text)
    parsed_text = parsed_text.split("\n\n")
    parsed_text = [paragraph.replace("\n", " ").strip() for paragraph in parsed_text]

    return parsed_text

# this function only fixs number misreading error (e.g.,  4.4 -> 44 or 4.4.4 -> 44.4)
def try_simple_ocr_num_fix(text, matched_pattern):

    if matched_pattern in [ArabicPattern, SubArabicPattern]:
    # this is super hacky (based on the assumption that only the dot after the first number is missing)
        if text[0] != "1" and text[1] not in [" ", "."]:
            text = text[0] + "." + text[1:]
            if matched_pattern == ArabicPattern:
                matched_pattern = SubArabicPattern
            if matched_pattern == SubArabicPattern and text[3] not in [" ", "."]:
                text = text[:3] + "." + text[3:]
                matched_pattern = SubSubArabicPattern
            print("fix an ocr problem at: " + text)

    return text, matched_pattern

# the function will modify global_pattern_hierachy and root_node
def match_top_level_patterns(text, top_level_match_results, root_node):
    
    matched_top_pattern = AllowedTopLevelPatterns[np.where(top_level_match_results)[0][0]]
    
    if len(root_node.child_nodes) != 0:
        top_level_pattern = root_node.child_nodes[-1].pattern
        if top_level_pattern != OathPattern and top_level_pattern != matched_top_pattern:
            print("Conflicting top level patterns! Current Node: " + text)
            print("Previous Top-Level Node: " + root_node.child_nodes[-1].text)

    root_node.adopt_children([Node(matched_top_pattern, text)])
    
    return root_node

# the function will modify pattern_hierachy, prev_node and parent_node
def match_sub_level_patterns(text, sub_level_match_results, pattern_hierachy,
                             root_node, prev_node, parent_node):

    matched_sub_pattern = AllowedSubLevelPatterns[np.where(sub_level_match_results)[0][0]]

    text, matched_sub_pattern = try_simple_ocr_num_fix(text, matched_sub_pattern)
    
    if matched_sub_pattern not in pattern_hierachy:
        pattern_hierachy.append(matched_sub_pattern)
                       
    prev_lv = pattern_hierachy.index(prev_node.pattern)
    matched_lv = pattern_hierachy.index(matched_sub_pattern)
    lv_diff = prev_lv - matched_lv

    if lv_diff > 0:
        # move up
        parent_node = find_parent_node(root_node, matched_sub_pattern)
        
    elif lv_diff < 0:                    
        # move down
        parent_node = prev_node

    prev_node = Node(matched_sub_pattern, text)
    parent_node.adopt_children([prev_node])
    
    return prev_node, parent_node
    
def build_text_tree(raw_text):
    
    paragraphs = pre_clean(raw_text)
    
    root_node = Node("","")
    curr_pattern_hierachy = []
    parent_node = root_node
    prev_node = None
    
    for idx, paragraph in enumerate(paragraphs):
        if re.match(OathPattern, paragraph) is not None:
            oath_node = Node(OathPattern, paragraph)
            root_node.adopt_children([oath_node])
            prev_node = oath_node
            parent_node = oath_node
            curr_pattern_hierachy=[OathPattern]
            continue

        # match top-level patterns
        top_level_match_results = [re.match(pattern, paragraph) is not None 
                                   for pattern in AllowedTopLevelPatterns]
        if any(top_level_match_results):
            root_node = match_top_level_patterns(paragraph, top_level_match_results, root_node)
            prev_node = root_node.child_nodes[-1]
            parent_node = root_node.child_nodes[-1]
            curr_pattern_hierachy=[parent_node.pattern]
            continue

        # skip intro section
        if len(root_node.child_nodes) == 0:
            continue

        # match sub-level patterns
        sub_level_match_results = [re.match(pattern, paragraph) is not None 
                                   for pattern in AllowedSubLevelPatterns]
        if any(sub_level_match_results):
            
            prev_node, parent_node = match_sub_level_patterns(paragraph, sub_level_match_results, curr_pattern_hierachy, 
                                                 root_node, prev_node, parent_node)
            continue

        # match nothing
        prev_node.append_text(paragraph)
    
    root_node.deep_clean()

    return root_node
