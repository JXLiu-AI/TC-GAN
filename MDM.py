
from init import guidedfilter
import time
from PIL import Image
import cv2 as cv
import matplotlib.image as mpimg  
import scipy.misc as misc
import cv2
import numpy as np
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
img = Image.open(r'E:\Dataset\results_1\epoch_1_index_3.png')
Img = img.convert('L')
Img.save("test1.jpg")
image1 = mpimg.imread(r'E:\Project\EvaluationCode\IRVI\RoadSences\gray_IR\IR39.jpg')            
image2 = mpimg.imread(r'E:\Project\EvaluationCode\IRVI\RoadSences\gray_VIS\VIS39.jpg');          

guid = mpimg.imread(r'E:\Project\NewProject\ResNetFusion-master\test1.jpg');

start = time.time()
new = guidedfilter((guid / 255.0),(image1 / 255.0),5, 0.01)               

new = (new - np.min(new)) / (np.max(new) - np.min(new))
misc.imsave('final_fusion1.jpg', new)
new1 = guidedfilter((guid / 255.0), (image2/ 255.0) ,5, 0.01)              

new1 = (new1 - np.min(new1)) / (np.max(new1) - np.min(new1))
misc.imsave('final_fusion2.jpg', new1)


fusion = np.add(np.multiply(image1, new), np.multiply(image2, 1 - new))
fusion1 = np.add(np.multiply(image1, new1), np.multiply(image2, 1 - new1))
misc.imsave(r'E:\Dataset\results/r1.png', fusion)
misc.imsave(r'E:\Dataset\results/r2.png', fusion1)
fusion = 0.5 * fusion + 0.5 * fusion1
end = time.time()
print(end - start)
misc.imsave('AAAAA.png', fusion)                                            





