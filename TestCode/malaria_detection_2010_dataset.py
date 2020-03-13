from custom_classes import path, cv_iml


class Malaria_images:
    def __init__(self, file_name, type, x1, y1, x2, y2, x3, y3):
        self.file_name = file_name
        self.type = type
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3


annotation_file_path = path.dataset_path + "Malaria_dataset/malaria.txt"
fp = open(annotation_file_path, 'r')
file_line_by_line = []
for x in fp:
  file_index_start = x.find('file', 2)
  file_index_end = x.find('file', file_index_start +2)
  print()
  file_line_by_line.append(x)

#%%


