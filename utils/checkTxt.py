import os
from xml2txt import convert_annotation
path = "./labels"
txtfilename = os.listdir(path)
for i in txtfilename:
    filePath = os.path.join(path, i)
    size1 = os.path.getsize(filePath)
    if size1 == 0:
        print(i)
        # name = i.split('.')[0]
        # convert_annotation(name)

    # with open(filePath, 'r') as f:
    #     for eachline in f:
    #         eachline = eachline.strip()
    #         # print(eachline)
    #         eachline = eachline.split(' ')
    #         # print(eachline)
    #         if len(eachline) == 0:
    #             print(i)