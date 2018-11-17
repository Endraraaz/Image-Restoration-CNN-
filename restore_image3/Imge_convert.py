# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt


# new_im = plt.imread("Mozill.png")[:,:,:3]
# im2 = Image.fromarray((new_im*255).astype(np.uint8))

# im2.save('foo.png')



# To Crop
from PIL import Image
new_width,new_height=100,100
im = Image.open("zebra.jpg")
width, height = im.size   # Get dimensions

left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

im=im.crop((left, top, right, bottom))
im.save('zebra_GT.png')


# To convert  RGBA to RGB

# from PIL import Image

# png = Image.open("lena.png")
# png.load() # required for png.split()
# # new_im = plt.imread("image.png")[:,:,:3]
# background = Image.new("RGB", png.size, (255, 255, 255))
# background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

# background.save('lena.png', 'PNG', quality=100)


# To convert JPG to PNG

# from PIL import Image

# im = Image.open('forest.jpg')
# im.save('blurred.png')