import os
import pandas as pd
from custom_classes import path


def load_train_test_val_images_from(folder_path, file_extension=".JPG"):
    # This function load image for hard negative mining experiments. from train test and validation folder.
    # more simple form of this function is already implemented in malaria_detction_main.

    train_dir = os.path.join(folder_path, 'train')
    test_dir = os.path.join(folder_path, 'test')
    validation_dir = os.path.join(folder_path, 'val')

    # %%
    # train data
    train_infected_dir = train_dir + "/Parasitized/"
    train_healthy_dir = train_dir + "/Uninfected/"

    train_infected_dir = [train_infected_dir + sub for sub in path.read_all_files_name_from(train_infected_dir, file_extension)]
    train_healthy_dir = [train_healthy_dir + sub for sub in path.read_all_files_name_from(train_healthy_dir, file_extension)]

    files_df = pd.DataFrame({
        'filename': train_infected_dir + train_healthy_dir,
        'label': ['malaria'] * len(train_infected_dir) + ['healthy'] * len(train_healthy_dir)
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    # Generating tanning and testing data.
    train_files = files_df['filename'].values
    train_labels = files_df['label'].values
#%%
    # test data
    test_infected_dir = os.path.join(test_dir, "Parasitized/")
    test_healthy_dir = os.path.join(test_dir, "Uninfected/")

    test_infected_dir = [test_infected_dir + sub for sub in path.read_all_files_name_from(test_infected_dir, file_extension)]
    test_healthy_dir = [test_healthy_dir + sub for sub in path.read_all_files_name_from(test_healthy_dir, file_extension)]

    files_df = pd.DataFrame({
        'filename': test_infected_dir + test_healthy_dir,
        'label': ['malaria'] * len(test_infected_dir) + ['healthy'] * len(test_healthy_dir)
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    # Generating tanning and testing data.
    test_files = files_df['filename'].values
    test_labels = files_df['label'].values

#%%
    # validation data
    val_infected_dir = os.path.join(validation_dir, "Parasitized/")
    val_healthy_dir = os.path.join(validation_dir, "Uninfected/")

    val_infected_dir = [val_infected_dir + sub for sub in path.read_all_files_name_from(val_infected_dir, file_extension)]
    val_healthy_dir = [val_healthy_dir + sub for sub in path.read_all_files_name_from(val_healthy_dir, file_extension)]

    files_df = pd.DataFrame({
        'filename': val_infected_dir + val_healthy_dir,
        'label': ['malaria'] * len(val_infected_dir) + ['healthy'] * len(val_healthy_dir)
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    # Generating tanning and testing data.
    val_files = files_df['filename'].values
    val_labels = files_df['label'].values
    # %%

    return train_files, train_labels, test_files, test_labels,  val_files, val_labels
