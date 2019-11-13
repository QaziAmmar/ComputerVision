# This file contain the paths to all required dataset, saved model and other reults
import sys

def init():
    global base_path
    global dataset_path
    global save_models_path
    global download_path
    global result_folder_path

    if sys.platform == ("linux1" or "linux2"):
        base_path = "/home/ali/Desktop/ammar/Thesis/Model_Result_Dataset/"
    elif sys.platform == "darwin":
        base_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/"
    
    dataset_path = base_path + "Dataset/"
    save_models_path = base_path + "SavedModel/"
    download_path = "/Users/qaziammar/Downloads/"
    result_folder_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Results/"


