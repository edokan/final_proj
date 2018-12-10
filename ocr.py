from output import *
from segmentation.segment_document import *

def write_doc_to_file(document_path, name):
    segmented_doc = segment_document(document_path)
    write_output_file(segmented_doc, name)
    

def main():
    try:
        document_path = sys.argv[1]
        name = sys.argv[2]
    except IndexError:
        print("Please provice document path, output file name")
        return
    print("start:")
    write_doc_to_file(document_path, name)
    print("done!")

main()