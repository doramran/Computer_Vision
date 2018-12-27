from PIL import Image,ImageDraw
from cv_project import Buses_data_opening



path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/computer_vision/final_project'
data,labels = Buses_data_opening(path)
image = data[0]
rects = labels[0]

for i in range(len(rects)):
    print(rects[i])
    del rects[i][4]
    print(rects[i])

image = Image.fromarray(image)
draw1 = ImageDraw.Draw(image)
draw2 = ImageDraw.Draw(image)
draw1.rectangle(rects[0],fill = None,outline = 'red')
draw2.rectangle(rects[1],fill = None,outline = 'red')
image.show()
