# This code parses date/times, so please
#
#     pip install python-dateutil
#
# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = welcome_from_dict(json.loads(json_string))

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
    MSCS18031_ITU_EDU_PK = "mscs18031@itu.edu.pk"
    MUAAZ_ZAKRIA_ITU_EDU_PK = "muaaz.zakria@itu.edu.pk"
    SYED_JAVED_ITU_EDU_PK = "syed.javed@itu.edu.pk"
    USAMA_IRFAN_ITU_EDU_PK = "usama.irfan@itu.edu.pk"


class DatasetName(Enum):
    SHALAMAR_LOCALIZATION = "Shalamar Localization "


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
    THE_1_CE6_FF = "#1CE6FF"


class SchemaID(Enum):
    CKGHYMONU1_CJY0_Y94_BC5_TB10_S = "ckghymonu1cjy0y94bc5tb10s"


class Title(Enum):
    HEALTHY = "healthy"


class Object:
    feature_id: str
    schema_id: SchemaID
    title: Title
    value: Title
    color: Color
    bbox: Bbox
    instance_uri: str

    def __init__(self, feature_id: str, schema_id: SchemaID, title: Title, value: Title, color: Color, bbox: Bbox, instance_uri: str) -> None:
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
        value = Title(obj.get("value"))
        color = Color(obj.get("color"))
        bbox = Bbox.from_dict(obj.get("bbox"))
        instance_uri = from_str(obj.get("instanceURI"))
        return Object(feature_id, schema_id, title, value, color, bbox, instance_uri)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureId"] = from_str(self.feature_id)
        result["schemaId"] = to_enum(SchemaID, self.schema_id)
        result["title"] = to_enum(Title, self.title)
        result["value"] = to_enum(Title, self.value)
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
    SHALAMAR_BOUNDING_BOXES_LOCALIZATION = "Shalamar Bounding Boxes localization"


class Review:
    score: int
    id: str
    created_at: datetime
    created_by: CreatedBy

    def __init__(self, score: int, id: str, created_at: datetime, created_by: CreatedBy) -> None:
        self.score = score
        self.id = id
        self.created_at = created_at
        self.created_by = created_by

    @staticmethod
    def from_dict(obj: Any) -> 'Review':
        assert isinstance(obj, dict)
        score = from_int(obj.get("score"))
        id = from_str(obj.get("id"))
        created_at = from_datetime(obj.get("createdAt"))
        created_by = CreatedBy(obj.get("createdBy"))
        return Review(score, id, created_at, created_by)

    def to_dict(self) -> dict:
        result: dict = {}
        result["score"] = from_int(self.score)
        result["id"] = from_str(self.id)
        result["createdAt"] = self.created_at.isoformat()
        result["createdBy"] = to_enum(CreatedBy, self.created_by)
        return result


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
    agreement: Optional[int]
    benchmark_agreement: int
    benchmark_id: None
    dataset_name: DatasetName
    reviews: List[Review]
    view_label: str

    def __init__(self, id: str, data_row_id: str, labeled_data: str, label: Label, created_by: CreatedBy, project_name: ProjectName, created_at: datetime, updated_at: datetime, seconds_to_label: float, external_id: str, agreement: Optional[int], benchmark_agreement: int, benchmark_id: None, dataset_name: DatasetName, reviews: List[Review], view_label: str) -> None:
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
        agreement = from_union([from_none, from_int], obj.get("Agreement"))
        benchmark_agreement = from_int(obj.get("Benchmark Agreement"))
        benchmark_id = from_none(obj.get("Benchmark ID"))
        dataset_name = DatasetName(obj.get("Dataset Name"))
        reviews = from_list(Review.from_dict, obj.get("Reviews"))
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
        result["Agreement"] = from_union([from_none, from_int], self.agreement)
        result["Benchmark Agreement"] = from_int(self.benchmark_agreement)
        result["Benchmark ID"] = from_none(self.benchmark_id)
        result["Dataset Name"] = to_enum(DatasetName, self.dataset_name)
        result["Reviews"] = from_list(lambda x: to_class(Review, x), self.reviews)
        result["View Label"] = from_str(self.view_label)
        return result


def welcome_from_dict(s: Any) -> List[WelcomeElement]:
    return from_list(WelcomeElement.from_dict, s)


def welcome_to_dict(x: List[WelcomeElement]) -> Any:
    return from_list(lambda x: to_class(WelcomeElement, x), x)
