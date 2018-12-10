import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
import PIL
import time

WHITE = (255, 255, 255)

def segment_lines(threshed, original_doc):
    horizontal_hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    th = 0
    H,W = threshed.shape[:2]
    uppers = [y for y in range(H-1) if horizontal_hist[y]<=th and horizontal_hist[y+1]>th]
    lowers = [y for y in range(H-1) if horizontal_hist[y]>th and horizontal_hist[y+1]<=th]
    assert len(uppers) == len(lowers), "Could not segment lines."

    lines = []
    original_lines = []
    for i in range(len(uppers)):
        cropped = threshed[uppers[i]:lowers[i], 0:W].copy()
        original = original_doc[uppers[i]:lowers[i], 0:W].copy()
        lines.append(cropped)
        original_lines.append(original)
    return lines, original_lines

def segment_words(lines, asarray, original_lines):
    segmented_lines = []
    for l in range(len(lines)):
        vertical_hist = cv2.reduce(lines[l], 0, cv2.REDUCE_AVG).reshape(-1)
        th = 0
        H,W = lines[l].shape[:2]
        lefts = [x for x in range(W-1) if vertical_hist[x]<=th and vertical_hist[x+1]>th]
        rights = [x for x in range(W-1) if vertical_hist[x]>th and vertical_hist[x+1]<=th]
        assert len(lefts) == len(rights), "Could not segment words"

        gaps = [lefts[i+1] - rights[i] for i in range(len(lefts)-1)]
        gaps = np.array(gaps).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(gaps)
        gap_labels = kmeans.labels_

        n_segments = len(lefts)
        word_gap_i, char_gap_i = np.argmax(gaps), np.argmin(gaps)
        word_label, char_label = gap_labels[word_gap_i], gap_labels[char_gap_i]

        segmented_line = []
        word = []
        for i in range(n_segments):
            if i != 0:
                if gap_labels[i-1] == word_label:
                    segmented_line.append(word)
                    word = []
            cropped_char = original_lines[l][0:H, lefts[i]:rights[i]].copy()
            char_h, char_w = cropped_char.shape[:2]
            pad_factor = char_h//32 - char_w//32
            extra_pad = abs(32*pad_factor)
            delta_h = 32 - (char_h % 32)
            delta_w = 32 - (char_w % 32)
            if pad_factor > 0:
                delta_w += extra_pad
            elif pad_factor < 0:
                delta_h += extra_pad

            pad_top = delta_h // 2 + (delta_h % 2 == 1)
            pad_bot = delta_h // 2

            pad_left = delta_w // 2 + (delta_w % 2 == 1)
            pad_right = delta_w // 2

            cropped_char = cv2.copyMakeBorder(cropped_char,pad_top,pad_bot,pad_left,pad_right,cv2.BORDER_CONSTANT,value=WHITE)
            if not asarray:
                cropped_char = PIL.Image.fromarray(cropped_char)
                cropped_char = cropped_char.resize((32, 32), PIL.Image.BILINEAR)
            else:
                cv2.resize(cropped_char, (32, 32))
            word.append(cropped_char)
        segmented_line.append(word)

        segmented_lines.append(segmented_line)

    return segmented_lines

    


def segment_document(document_path, asarray=False):
    """
    Input: document_path (str)- path to image to be segmented

    Returns: document- [lines[words[chars]]],
        where char is an image of a character
    """
    # Read image
    doc = cv2.imread(document_path)

    # Convert to grayscale
    gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

    # Binarize by threshold and invert
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Segment lines
    lines, original_lines = segment_lines(threshed, gray)

    # Segment words
    lines_by_words = segment_words(lines, asarray, original_lines)

    return lines_by_words


def main():
    try:
        document_path = sys.argv[1]
    except IndexError:
        print("Did not provide path to document!")
        return
    segmented_doc = segment_document(document_path, False)
    p = True
    for i in range(len(segmented_doc)):
        print("line {} :".format(i+1))
        for j in range(len(segmented_doc[i])):
            print("word " + str(j+1) + " length: " + str(len(segmented_doc[i][j])))
            for c in segmented_doc[i][j]:
                if p:
                    c.show()
            p = False
                # cv2.imshow("c", c)
                # cv2.waitKey(0)

if __name__ == '__main__':
    main()
