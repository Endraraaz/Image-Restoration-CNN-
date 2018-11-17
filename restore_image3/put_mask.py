
from PIL import  ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import os
from PIL import Image

width = 100
height = 100
center = height//2
white = (255, 255, 255)
green = (0,128,0)

path_to_image,output_path=list(sys.argv)[1],list(sys.argv)[2]

def save():
    filename = "image.png"
    image1.save(filename)
    filename='image_mask.png'
    image2.save(filename)
    # cv2.postscript(file="image_mask.eps")
    # img = PIL.Image.open("image_mask.eps")
    # img.save("image_mask.png", "png")
    os.system('python neural-inpainting.py'+" "+output_path)
def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)
    draw2.line([x1, y1, x2, y2],fill="black",width=5)
root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')

cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
if 'jpg' in path_to_image:

    im = Image.open(path_to_image)
    path_to_image=path_to_image.replace("jpg","png")
    im.save(path_to_image)
image1 = PIL.Image.open(path_to_image)
draw = ImageDraw.Draw(image1)

image2 = PIL.Image.new("RGB", (image1.size[0], image1.size[1]), white)
draw2 = ImageDraw.Draw(image2)

pic = PhotoImage(file = path_to_image)
# print(type(pic))
cv.create_image(0, 0, image = pic, anchor = NW)
# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)
button=Button(text="save",command=save)
button.pack()
root.mainloop()
