
import numpy
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def calculateDistance(i1, i2):
    return numpy.sum((i1-i2)**2)

test1 = cv2.imread("/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/Shalamar_Captured_Malaria/background_img.JPG")
test2 = cv2.imread("/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/Shalamar_Captured_Malaria/PA171818_8.JPG")
test3 = cv2.imread("/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/Shalamar_Captured_Malaria/PA171855_49.JPG")
test4 = cv2.imread("/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/Shalamar_Captured_Malaria/PA172017_19.JPG")

test1 = cv2.resize(test1, (68, 68)) / 255.
test2 = cv2.resize(test2, (68, 68)) / 255.
test3 = cv2.resize(test3, (68, 68)) / 255.
test4 = cv2.resize(test4, (68, 68)) / 255.



# test1 = numpy.reshape(test1, (-1, 1))
# test2 = numpy.reshape(test2, (-1, 1))
# test3 = numpy.reshape(test3, (-1, 1))

# diff imgs
dist1 = calculateDistance(test1, test2)
# same imgs
dist2 = calculateDistance(test1, test3)
# same diff slide
dist3 = calculateDistance(test1, test4)
