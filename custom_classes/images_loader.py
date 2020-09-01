import os
import pandas as pd
import glob
import pathlib


def process_path(data_dir, file_extension=".JPG"):
    """
    This function go through the entire file path and return all files in that folder.
    :param data_dir: directory of data where you want to load all image
    :param file_extension:
    :return: return files path and labels of each files in pandas frame.
    """
    data_dir = pathlib.Path(data_dir)
    files_df = None
    for folder_name in data_dir.glob('*'):
        # '.DS_Store' file is automatically created by mac which we need to exclude form your code.
        if '.DS_Store' == str(folder_name).split('/')[-1]:
            continue
        files_in_folder = glob.glob(str(folder_name) + '/*' + file_extension)
        df2 = pd.DataFrame({
            'filename': files_in_folder,
            'label': [folder_name.name] * len(files_in_folder)
        })

        if files_df is None:
            files_df = df2
        else:
            files_df = files_df.append(df2, ignore_index=True)
    # to shuffle the data
    files_df = files_df.sample(frac=1).reset_index(drop=True)

    files = files_df['filename'].values
    labels = files_df['label'].values

    return files, labels


def load_train_test_val_images_from(folder_path, file_extension=".JPG"):
    """

    :param folder_path:
    :param file_extension:
    :return:
    """

    train_dir = os.path.join(folder_path, 'train')
    test_dir = os.path.join(folder_path, 'test')
    validation_dir = os.path.join(folder_path, 'val')

    train_files, train_labels = process_path(train_dir, file_extension)
    test_files, test_labels = process_path(test_dir, file_extension)
    val_files, val_labels = process_path(validation_dir, file_extension)

    return train_files, train_labels, test_files, test_labels, val_files, val_labels
