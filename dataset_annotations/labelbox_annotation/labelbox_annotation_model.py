# from enum import Enum
# from dataclasses import dataclass
# from typing import Any, Optional, List, TypeVar, Type, cast, Callable
# from datetime import datetime
# import dateutil.parser
#
#
# T = TypeVar("T")
# EnumT = TypeVar("EnumT", bound=Enum)
#
#
# def from_int(x: Any) -> int:
#     assert isinstance(x, int) and not isinstance(x, bool)
#     return x
#
#
# def from_str(x: Any) -> str:
#     assert isinstance(x, str)
#     return x
#
#
# def to_enum(c: Type[EnumT], x: Any) -> EnumT:
#     assert isinstance(x, c)
#     return x.value
#
#
# def to_class(c: Type[T], x: Any) -> dict:
#     assert isinstance(x, c)
#     return cast(Any, x).to_dict()
#
#
# def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
#     assert isinstance(x, list)
#     return [f(y) for y in x]
#
#
# def from_none(x: Any) -> Any:
#     assert x is None
#     return x
#
#
# def from_union(fs, x):
#     for f in fs:
#         try:
#             return f(x)
#         except:
#             pass
#     assert False
#
#
# def from_datetime(x: Any) -> datetime:
#     return dateutil.parser.parse(x)
#
#
# def from_float(x: Any) -> float:
#     assert isinstance(x, (float, int)) and not isinstance(x, bool)
#     return float(x)
#
#
# def to_float(x: Any) -> float:
#     assert isinstance(x, float)
#     return x
#
#
# class CreatedBy(Enum):
#     MSCS18006_ITU_EDU_PK = "mscs18006@itu.edu.pk"
#     MSCS18031_ITU_EDU_PK = "mscs18031@itu.edu.pk"
#     NADEEM_YOUSAF_ITU_EDU_PK = "nadeem.yousaf@itu.edu.pk"
#     WASEEM_ASHRAF_ITU_EDU_PK = "waseem.ashraf@itu.edu.pk"
#
#
# class DatasetName(Enum):
#     P_VIVAX_100_X_CROP = "P.Vivax_100x_crop"
#
#
# @dataclass
# class Bbox:
#     top: int
#     left: int
#     height: int
#     width: int
#
#     @staticmethod
#     def from_dict(obj: Any) -> 'Bbox':
#         assert isinstance(obj, dict)
#         top = from_int(obj.get("top"))
#         left = from_int(obj.get("left"))
#         height = from_int(obj.get("height"))
#         width = from_int(obj.get("width"))
#         return Bbox(top, left, height, width)
#
#     def to_dict(self) -> dict:
#         result: dict = {}
#         result["top"] = from_int(self.top)
#         result["left"] = from_int(self.left)
#         result["height"] = from_int(self.height)
#         result["width"] = from_int(self.width)
#         return result
#
#
# class Color(Enum):
#     FF0000 = "#FF0000"
#     FF8000 = "#FF8000"
#
#
# class SchemaID(Enum):
#     CKBKL1_G750_O670_Y6_D9_Z2_G5952 = "ckbkl1g750o670y6d9z2g5952"
#     CKBKL1_G750_O680_Y6_D6_SXTBT97 = "ckbkl1g750o680y6d6sxtbt97"
#
#
# class Title(Enum):
#     MALARIA_RING = "Malaria - Ring"
#     RED_BLOOD_CELL_HEALTHY = "Red Blood Cell - Healthy"
#
#
# class Value(Enum):
#     MALARIA_RING = "malaria_-_ring"
#     RED_BLOOD_CELL_HEALTHY = "red_blood_cell_-_healthy"
#
#
# @dataclass
# class Object:
#     feature_id: str
#     schema_id: SchemaID
#     title: Title
#     value: Value
#     color: Color
#     bbox: Bbox
#     instance_uri: str
#
#     @staticmethod
#     def from_dict(obj: Any) -> 'Object':
#         assert isinstance(obj, dict)
#         feature_id = from_str(obj.get("featureId"))
#         schema_id = SchemaID(obj.get("schemaId"))
#         title = Title(obj.get("title"))
#         value = Value(obj.get("value"))
#         color = Color(obj.get("color"))
#         bbox = Bbox.from_dict(obj.get("bbox"))
#         instance_uri = from_str(obj.get("instanceURI"))
#         return Object(feature_id, schema_id, title, value, color, bbox, instance_uri)
#
#     def to_dict(self) -> dict:
#         result: dict = {}
#         result["featureId"] = from_str(self.feature_id)
#         result["schemaId"] = to_enum(SchemaID, self.schema_id)
#         result["title"] = to_enum(Title, self.title)
#         result["value"] = to_enum(Value, self.value)
#         result["color"] = to_enum(Color, self.color)
#         result["bbox"] = to_class(Bbox, self.bbox)
#         result["instanceURI"] = from_str(self.instance_uri)
#         return result
#
#
# @dataclass
# class Label:
#     objects: Optional[List[Object]] = None
#     classifications: Optional[List[Any]] = None
#
#     @staticmethod
#     def from_dict(obj: Any) -> 'Label':
#         assert isinstance(obj, dict)
#         objects = from_union([lambda x: from_list(Object.from_dict, x), from_none], obj.get("objects"))
#         classifications = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("classifications"))
#         return Label(objects, classifications)
#
#     def to_dict(self) -> dict:
#         result: dict = {}
#         result["objects"] = from_union([lambda x: from_list(lambda x: to_class(Object, x), x), from_none], self.objects)
#         result["classifications"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.classifications)
#         return result
#
#
# class ProjectName(Enum):
#     P_VIVAX_100_X_CROP = "P.vivax_100xCrop"
#
#
# @dataclass
# class WelcomeElement:
#     id: str
#     data_row_id: str
#     labeled_data: str
#     label: Label
#     created_by: CreatedBy
#     project_name: ProjectName
#     created_at: datetime
#     updated_at: datetime
#     seconds_to_label: float
#     external_id: str
#     agreement: int
#     benchmark_agreement: int
#     benchmark_id: None
#     dataset_name: DatasetName
#     reviews: List[Any]
#     view_label: str
#
#     @staticmethod
#     def from_dict(obj: Any) -> 'WelcomeElement':
#         assert isinstance(obj, dict)
#         id = from_str(obj.get("ID"))
#         data_row_id = from_str(obj.get("DataRow ID"))
#         labeled_data = from_str(obj.get("Labeled Data"))
#         label = Label.from_dict(obj.get("Label"))
#         created_by = CreatedBy(obj.get("Created By"))
#         project_name = ProjectName(obj.get("Project Name"))
#         created_at = from_datetime(obj.get("Created At"))
#         updated_at = from_datetime(obj.get("Updated At"))
#         seconds_to_label = from_float(obj.get("Seconds to Label"))
#         external_id = from_str(obj.get("External ID"))
#         agreement = from_int(obj.get("Agreement"))
#         benchmark_agreement = from_int(obj.get("Benchmark Agreement"))
#         benchmark_id = from_none(obj.get("Benchmark ID"))
#         dataset_name = DatasetName(obj.get("Dataset Name"))
#         reviews = from_list(lambda x: x, obj.get("Reviews"))
#         view_label = from_str(obj.get("View Label"))
#         return WelcomeElement(id, data_row_id, labeled_data, label, created_by, project_name, created_at, updated_at, seconds_to_label, external_id, agreement, benchmark_agreement, benchmark_id, dataset_name, reviews, view_label)
#
#     def to_dict(self) -> dict:
#         result: dict = {}
#         result["ID"] = from_str(self.id)
#         result["DataRow ID"] = from_str(self.data_row_id)
#         result["Labeled Data"] = from_str(self.labeled_data)
#         result["Label"] = to_class(Label, self.label)
#         result["Created By"] = to_enum(CreatedBy, self.created_by)
#         result["Project Name"] = to_enum(ProjectName, self.project_name)
#         result["Created At"] = self.created_at.isoformat()
#         result["Updated At"] = self.updated_at.isoformat()
#         result["Seconds to Label"] = to_float(self.seconds_to_label)
#         result["External ID"] = from_str(self.external_id)
#         result["Agreement"] = from_int(self.agreement)
#         result["Benchmark Agreement"] = from_int(self.benchmark_agreement)
#         result["Benchmark ID"] = from_none(self.benchmark_id)
#         result["Dataset Name"] = to_enum(DatasetName, self.dataset_name)
#         result["Reviews"] = from_list(lambda x: x, self.reviews)
#         result["View Label"] = from_str(self.view_label)
#         return result
#
#
# def welcome_from_dict(s: Any) -> List[WelcomeElement]:
#     return from_list(WelcomeElement.from_dict, s)
#
#
# def welcome_to_dict(x: List[WelcomeElement]) -> Any:
#     return from_list(lambda x: to_class(WelcomeElement, x), x)
#
#

# PF MODEL generator

from enum import Enum
from typing import Any, Optional, List, TypeVar, Type, cast, Callable
from datetime import datetime
import dateutil.parser


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_datetime(x: Any) -> datetime:
    return dateutil.parser.parse(x)


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


class CreatedBy(Enum):
    MSCS18006_ITU_EDU_PK = "mscs18006@itu.edu.pk"
    MSCS18031_ITU_EDU_PK = "mscs18031@itu.edu.pk"
    NADEEM_YOUSAF_ITU_EDU_PK = "nadeem.yousaf@itu.edu.pk"
    WASEEM_ASHRAF_ITU_EDU_PK = "waseem.ashraf@itu.edu.pk"


class DatasetName(Enum):
    P_F_100_X_CROP = "p.f_100xCrop"


class Bbox:
    top: int
    left: int
    height: int
    width: int

    def __init__(self, top: int, left: int, height: int, width: int) -> None:
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    @staticmethod
    def from_dict(obj: Any) -> 'Bbox':
        assert isinstance(obj, dict)
        top = from_int(obj.get("top"))
        left = from_int(obj.get("left"))
        height = from_int(obj.get("height"))
        width = from_int(obj.get("width"))
        return Bbox(top, left, height, width)

    def to_dict(self) -> dict:
        result: dict = {}
        result["top"] = from_int(self.top)
        result["left"] = from_int(self.left)
        result["height"] = from_int(self.height)
        result["width"] = from_int(self.width)
        return result


class Color(Enum):
    FF0000 = "#FF0000"


class SchemaID(Enum):
    CKBRU8_WFB0_N0_J0_Y6_BHZB3_G2_S2 = "ckbru8wfb0n0j0y6bhzb3g2s2"


class Title(Enum):
    RED_BLOOD_CELL_HEALTHY = "Red Blood Cell - Healthy"


class Value(Enum):
    RED_BLOOD_CELL_HEALTHY = "red_blood_cell_-_healthy"


class Object:
    feature_id: str
    schema_id: SchemaID
    title: Title
    value: Value
    color: Color
    bbox: Bbox
    instance_uri: str

    def __init__(self, feature_id: str, schema_id: SchemaID, title: Title, value: Value, color: Color, bbox: Bbox, instance_uri: str) -> None:
        self.feature_id = feature_id
        self.schema_id = schema_id
        self.title = title
        self.value = value
        self.color = color
        self.bbox = bbox
        self.instance_uri = instance_uri

    @staticmethod
    def from_dict(obj: Any) -> 'Object':
        assert isinstance(obj, dict)
        feature_id = from_str(obj.get("featureId"))
        schema_id = SchemaID(obj.get("schemaId"))
        title = Title(obj.get("title"))
        value = Value(obj.get("value"))
        color = Color(obj.get("color"))
        bbox = Bbox.from_dict(obj.get("bbox"))
        instance_uri = from_str(obj.get("instanceURI"))
        return Object(feature_id, schema_id, title, value, color, bbox, instance_uri)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureId"] = from_str(self.feature_id)
        result["schemaId"] = to_enum(SchemaID, self.schema_id)
        result["title"] = to_enum(Title, self.title)
        result["value"] = to_enum(Value, self.value)
        result["color"] = to_enum(Color, self.color)
        result["bbox"] = to_class(Bbox, self.bbox)
        result["instanceURI"] = from_str(self.instance_uri)
        return result


class Label:
    objects: Optional[List[Object]]
    classifications: Optional[List[Any]]

    def __init__(self, objects: Optional[List[Object]], classifications: Optional[List[Any]]) -> None:
        self.objects = objects
        self.classifications = classifications

    @staticmethod
    def from_dict(obj: Any) -> 'Label':
        assert isinstance(obj, dict)
        objects = from_union([lambda x: from_list(Object.from_dict, x), from_none], obj.get("objects"))
        classifications = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("classifications"))
        return Label(objects, classifications)

    def to_dict(self) -> dict:
        result: dict = {}
        result["objects"] = from_union([lambda x: from_list(lambda x: to_class(Object, x), x), from_none], self.objects)
        result["classifications"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.classifications)
        return result


class ProjectName(Enum):
    P_F_100_X_CROP_SEGMENTATION = "p.f_100xCrop Segmentation"


class WelcomeElement:
    id: str
    data_row_id: str
    labeled_data: str
    label: Label
    created_by: CreatedBy
    project_name: ProjectName
    created_at: datetime
    updated_at: datetime
    seconds_to_label: float
    external_id: str
    agreement: float
    benchmark_agreement: int
    benchmark_id: None
    dataset_name: DatasetName
    reviews: List[Any]
    view_label: str

    def __init__(self, id: str, data_row_id: str, labeled_data: str, label: Label, created_by: CreatedBy, project_name: ProjectName, created_at: datetime, updated_at: datetime, seconds_to_label: float, external_id: str, agreement: float, benchmark_agreement: int, benchmark_id: None, dataset_name: DatasetName, reviews: List[Any], view_label: str) -> None:
        self.id = id
        self.data_row_id = data_row_id
        self.labeled_data = labeled_data
        self.label = label
        self.created_by = created_by
        self.project_name = project_name
        self.created_at = created_at
        self.updated_at = updated_at
        self.seconds_to_label = seconds_to_label
        self.external_id = external_id
        self.agreement = agreement
        self.benchmark_agreement = benchmark_agreement
        self.benchmark_id = benchmark_id
        self.dataset_name = dataset_name
        self.reviews = reviews
        self.view_label = view_label

    @staticmethod
    def from_dict(obj: Any) -> 'WelcomeElement':
        assert isinstance(obj, dict)
        id = from_str(obj.get("ID"))
        data_row_id = from_str(obj.get("DataRow ID"))
        labeled_data = from_str(obj.get("Labeled Data"))
        label = Label.from_dict(obj.get("Label"))
        created_by = CreatedBy(obj.get("Created By"))
        project_name = ProjectName(obj.get("Project Name"))
        created_at = from_datetime(obj.get("Created At"))
        updated_at = from_datetime(obj.get("Updated At"))
        seconds_to_label = from_float(obj.get("Seconds to Label"))
        external_id = from_str(obj.get("External ID"))
        agreement = from_float(obj.get("Agreement"))
        benchmark_agreement = from_int(obj.get("Benchmark Agreement"))
        benchmark_id = from_none(obj.get("Benchmark ID"))
        dataset_name = DatasetName(obj.get("Dataset Name"))
        reviews = from_list(lambda x: x, obj.get("Reviews"))
        view_label = from_str(obj.get("View Label"))
        return WelcomeElement(id, data_row_id, labeled_data, label, created_by, project_name, created_at, updated_at, seconds_to_label, external_id, agreement, benchmark_agreement, benchmark_id, dataset_name, reviews, view_label)

    def to_dict(self) -> dict:
        result: dict = {}
        result["ID"] = from_str(self.id)
        result["DataRow ID"] = from_str(self.data_row_id)
        result["Labeled Data"] = from_str(self.labeled_data)
        result["Label"] = to_class(Label, self.label)
        result["Created By"] = to_enum(CreatedBy, self.created_by)
        result["Project Name"] = to_enum(ProjectName, self.project_name)
        result["Created At"] = self.created_at.isoformat()
        result["Updated At"] = self.updated_at.isoformat()
        result["Seconds to Label"] = to_float(self.seconds_to_label)
        result["External ID"] = from_str(self.external_id)
        result["Agreement"] = to_float(self.agreement)
        result["Benchmark Agreement"] = from_int(self.benchmark_agreement)
        result["Benchmark ID"] = from_none(self.benchmark_id)
        result["Dataset Name"] = to_enum(DatasetName, self.dataset_name)
        result["Reviews"] = from_list(lambda x: x, self.reviews)
        result["View Label"] = from_str(self.view_label)
        return result


def welcome_from_dict(s: Any) -> List[WelcomeElement]:
    return from_list(WelcomeElement.from_dict, s)


def welcome_to_dict(x: List[WelcomeElement]) -> Any:
    return from_list(lambda x: to_class(WelcomeElement, x), x)