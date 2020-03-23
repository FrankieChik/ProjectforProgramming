import os

for path, dirs, files in os.walk ('/Users/hinmingfrankiechik/Desktop/text_files'):
    for file in files:
        print(os.path.join(path, file))

path = "/Users/hinmingfrankiechik/Desktop/text_files/2001"
files= os.listdir(path)
txts = []
for file in files:
    position = path+'/'+ file
    print (position)           
    with open(position, "r",encoding='utf-8') as f:           
        lines = f.readlines()   
        for line in lines:
            txts.append(line)
        f.close()
print (txts)
