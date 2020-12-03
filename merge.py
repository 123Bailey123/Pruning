import os
from PyPDF2 import PdfFileMerger

def pdf_merge(path):
    merger = PdfFileMerger()
    print (path)
    for filename in sorted(os.listdir(path)):
        print (filename)
        if filename.endswith(".pdf"):
            merger.append(path+filename)

    merger.write(path[:-1]+"_combined.pdf")
    merger.close()
