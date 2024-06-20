from PIL import Image
import os



dir = "output_dreamlora"
filelist = os.listdir("output_dreamlora")
count = 0
for file_id in filelist:
    count = count + 1
    for file in os.listdir(dir + "/" + file_id):
        filename = dir + "/" + file_id + "/" + file
        print(filename)
        with Image.open(filename) as img:
            img = img.convert('RGB')
            img = img.resize((512, 512))
            img.save(filename)
