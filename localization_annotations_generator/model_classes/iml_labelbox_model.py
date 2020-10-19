# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = welcome_from_dict(json.loads(json_string))

from enum import Enum
from typing import Any, List, TypeVar, Type, cast, Callable


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


class CreatedBy(Enum):
    MSCS18031_ITU_EDU_PK = "mscs18031@itu.edu.pk"


class DatasetName(Enum):
    CHUGHATI_P_F = "Chughati P.F"


class Bbox:
    x: int
    y: int
    h: int
    w: int

    def __init__(self, x: int, y: int, h: int, w: int) -> None:
        self.x = x
        self.y = y
        self.h = h
        self.w = w

    @staticmethod
    def from_dict(obj: Any) -> 'Bbox':
        assert isinstance(obj, dict)
        x = from_int(obj.get("x"))
        y = from_int(obj.get("y"))
        h = from_int(obj.get("h"))
        w = from_int(obj.get("w"))
        return Bbox(x, y, h, w)

    def to_dict(self) -> dict:
        result: dict = {}
        result["x"] = from_int(self.x)
        result["y"] = from_int(self.y)
        result["h"] = from_int(self.h)
        result["w"] = from_int(self.w)
        return result


class Category(Enum):
    RING = "ring"
    TROPHOZOITE = "trophozoite"


class Object:
    category: Category
    bbox: Bbox

    def __init__(self, category: Category, bbox: Bbox) -> None:
        self.category = category
        self.bbox = bbox

    @staticmethod
    def from_dict(obj: Any) -> 'Object':
        assert isinstance(obj, dict)
        category = Category(obj.get("category"))
        bbox = Bbox.from_dict(obj.get("bbox"))
        return Object(category, bbox)

    def to_dict(self) -> dict:
        result: dict = {}
        result["category"] = to_enum(Category, self.category)
        result["bbox"] = to_class(Bbox, self.bbox)
        return result


class WelcomeElement:
    image_id: str
    created_by: CreatedBy
    dataset_name: DatasetName
    objects: List[Object]

    def __init__(self, image_id: str, created_by: CreatedBy, dataset_name: DatasetName, objects: List[Object]) -> None:
        self.image_id = image_id
        self.created_by = created_by
        self.dataset_name = dataset_name
        self.objects = objects

    @staticmethod
    def from_dict(obj: Any) -> 'WelcomeElement':
        assert isinstance(obj, dict)
        image_id = from_str(obj.get("image_id"))
        created_by = CreatedBy(obj.get("created_by"))
        dataset_name = DatasetName(obj.get("dataset_name"))
        objects = from_list(Object.from_dict, obj.get("objects"))
        return WelcomeElement(image_id, created_by, dataset_name, objects)

    def to_dict(self) -> dict:
        result: dict = {}
        result["image_id"] = from_str(self.image_id)
        result["created_by"] = to_enum(CreatedBy, self.created_by)
        result["dataset_name"] = to_enum(DatasetName, self.dataset_name)
        result["objects"] = from_list(lambda x: to_class(Object, x), self.objects)
        return result


def welcome_from_dict(s: Any) -> List[WelcomeElement]:
    return from_list(WelcomeElement.from_dict, s)


def welcome_to_dict(x: List[WelcomeElement]) -> Any:
    return from_list(lambda x: to_class(WelcomeElement, x), x)
