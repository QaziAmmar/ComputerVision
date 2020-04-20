from custom_classes import path, cv_iml

folderPth = path.result_folder_path + "pvivax_cells_augmented/leukocyte/"

cv_iml.augment_image(folderPth, file_extension='.png', rotation=False, flipping=True)
