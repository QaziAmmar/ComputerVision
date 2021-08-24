
## Computer Vision, Intelligent Machine Labs (CVIML)

This repository provides examples and best practice guidelines for building computer vision systems. The goal of this repository is to build a comprehensive set of tools and examples to leverage common computer vision task. Rather than creating implementations from scratch, we draw from existing state-of-the-art libraries and build additional utility around loading image data, showing train and test data and evaluating models. 
We hope that these examples and utilities can significantly reduce the “time to market” by simplifying the experience from defining the business problem to development of solution by orders of magnitude. 


## Classes

List of class that can support in Computer Vision task.

| Class Name | Link |
| ------ | ------ |
| cv_iml | https://github.com/QaziAmmar/ComputerVision/blob/master/custom_classes/cv_iml.py|
| path | https://github.com/QaziAmmar/ComputerVision/blob/master/custom_classes/path.py |
| perdefine_models | https://github.com/QaziAmmar/ComputerVision/blob/master/custom_classes/predefine_models.py][PlDb |



## cv_iml
Function use in this class
-  ```def image_show(img)```: This function replace your 2 line code with single line to show image. Just call this function and pass image to it.

-  ```def show_train_images(train_data, train_labels):```  Function is use to show either train/test data is load correctly.

- ```def show_histogram(img, title=None):``` This function shows the histogram of image. It first check if image is gray scale or rgb and then plot histogram according to it

- ```def removeBlackRegion(img):```  This function remove the black region form image by cropping the largest contours form the image. Use this function to remove the black region in microscope images, that may cause problems.

- ```def generate_patches_of_image(input_folder_path="", out_folder_path="", annotation_file_path=None, patch_size=0, annotation_label="")``` This function is use to generate image patches. 


- ```def get_f1_score(actual_labels, preds_labels, binary_classifcation, pos_label="malaria", confusion_matrix_title=""):```  Calculate the F1 score for prediction. This method works for both binary class and multiclass. For binary class you have to mention pos_label and for multiclass pass pos_label =1
- 
   
- ```def augment_image(images_folder_path, file_extension, rotation=True, flipping=True):```  Function rotate images in target folder at 90, 180 and 270 and flip image at vertically and horizontally on base of condition save them into same folder. This function can be enhanced for saving augmented images into some other folder.

## path
Function use in this class
-  ```def read_all_files_name_from(folder_path, file_extension):```: This function reads all files name from give folder with specific extension. 

## perdefine_models
This class contains some state of the art network in TensorFlow. You just need to pass the weight file name and your model will be available for learning.  


## Installation

For Installation just go to the link mention is  ```Classes Section``` copy the function and start using it. 


