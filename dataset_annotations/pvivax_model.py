class Annotation_Model(object):
    def __init__(self, annotation):
        self.image = Image(annotation['image'])
        self.objects = []
        for json_object in annotation['objects']:
            self.objects.append(Objects(json_object))


class Image(object):
    def __init__(self, image):
        self.checksum = image["checksum"]
        self.path_name = image["pathname"]
        self.shape = Shape(image["shape"])


class Shape(object):
    def __init__(self, shape):
        self.r = shape['r']
        self.c = shape['c']
        self.channels = shape['channels']


class Objects(object):
    def __init__(self, objects):
        self.category = objects['category']
        self.bounding_box = Bounding_box(objects['bounding_box'])


class Bounding_box(object):
    def __init__(self, bounding_box):
        minimum = bounding_box['minimum']
        maximum = bounding_box['maximum']

        self.y1 = minimum['r']
        self.x1 = minimum['c']
        self.y2 = maximum['r']
        self.x2 = maximum['c']
