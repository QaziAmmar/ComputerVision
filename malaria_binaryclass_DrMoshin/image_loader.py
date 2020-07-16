import os
import glob
from custom_classes import path


def load_train_test_val_images_from(folder_path):
    # This function load image for hard negative mining experiments. from train test and validation folder.
    # more simple form of this function is already implemented in malaria_detction_main.

    # folder_path = "IML_training_data/binary_classifcation_HardNegative_mining/p.f"

    train_dir = os.path.join(folder_path, 'train')
    test_dir = os.path.join(folder_path, 'test')
    validation_dir = os.path.join(folder_path, 'val')

    # %%

    train_infected_dir = os.path.join(train_dir, "Parasitized")
    train_healthy_dir = os.path.join(train_dir, "Uninfected")

    test_infected_dir = os.path.join(test_dir, "Parasitized")
    test_healthy_dir = os.path.join(test_dir, "Uninfected")

    val_infected_dir = os.path.join(validation_dir, "Parasitized")
    val_healthy_dir = os.path.join(validation_dir, "Uninfected")

    # %%

    infected_files = glob.glob(train_infected_dir + '/*.JPG') + glob.glob(test_infected_dir + '/*.JPG') + \
                     glob.glob(val_infected_dir + '/*.JPG')
    healthy_files = glob.glob(train_healthy_dir + '/*.JPG') + glob.glob(test_healthy_dir + '/*.JPG') + \
                    glob.glob(val_healthy_dir + '/*.JPG')

    return infected_files, healthy_files