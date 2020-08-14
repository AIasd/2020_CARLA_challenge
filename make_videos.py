import skvideo.io
import numpy as np
from PIL import Image
from datetime import datetime


folder_path = ''

for sub_folder_name in os.listdir(folder_path):
    sub_folder = os.path.join(folder_path, sub_folder_name)
    if os.path.isdir(sub_folder):
        for filename in os.listdir(sub_folder):
            if filename.endswith(".png"):
                filepath = os.path.join(sub_folder, filename)
                img = Image.open(filepath)
                img_np = asarray(image)
                imgs.append(img_np)

outputdata = np.stack(imgs)
outputdata = outputdata.astype(np.uint8)


now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


writer = skvideo.io.FFmpegWriter(time_str+".mp4")
for i in range(outputdata.shape[0].):
        writer.writeFrame(outputdata[i, :, :, :])
writer.close()
