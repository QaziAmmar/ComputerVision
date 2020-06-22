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

from dataclasses import dataclass
from typing import Any, List, TypeVar, Type, cast, Callable
from datetime import datetime
import dateutil.parser


T = TypeVar("T")


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_datetime(x: Any) -> datetime:
    return dateutil.parser.parse(x)


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_none(x: Any) -> Any:
    assert x is None
    return x


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


@dataclass
class Bbox:
    top: int
    left: int
    height: int
    width: int

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


@dataclass
class Object:
    feature_id: str
    schema_id: str
    title: str
    value: str
    color: str
    bbox: Bbox
    instance_uri: str

    @staticmethod
    def from_dict(obj: Any) -> 'Object':
        assert isinstance(obj, dict)
        feature_id = from_str(obj.get("featureId"))
        schema_id = from_str(obj.get("schemaId"))
        title = from_str(obj.get("title"))
        value = from_str(obj.get("value"))
        color = from_str(obj.get("color"))
        bbox = Bbox.from_dict(obj.get("bbox"))
        instance_uri = from_str(obj.get("instanceURI"))
        return Object(feature_id, schema_id, title, value, color, bbox, instance_uri)

    def to_dict(self) -> dict:
        result: dict = {}
        result["featureId"] = from_str(self.feature_id)
        result["schemaId"] = from_str(self.schema_id)
        result["title"] = from_str(self.title)
        result["value"] = from_str(self.value)
        result["color"] = from_str(self.color)
        result["bbox"] = to_class(Bbox, self.bbox)
        result["instanceURI"] = from_str(self.instance_uri)
        return result


@dataclass
class Label:
    objects: List[Object]
    classifications: List[Any]

    @staticmethod
    def from_dict(obj: Any) -> 'Label':
        assert isinstance(obj, dict)
        objects = from_list(Object.from_dict, obj.get("objects"))
        classifications = from_list(lambda x: x, obj.get("classifications"))
        return Label(objects, classifications)

    def to_dict(self) -> dict:
        result: dict = {}
        result["objects"] = from_list(lambda x: to_class(Object, x), self.objects)
        result["classifications"] = from_list(lambda x: x, self.classifications)
        return result


@dataclass
class WelcomeElement:
    id: str
    data_row_id: str
    labeled_data: str
    label: Label
    created_by: str
    project_name: str
    created_at: datetime
    updated_at: datetime
    seconds_to_label: float
    external_id: str
    agreement: int
    benchmark_agreement: int
    benchmark_id: None
    dataset_name: str
    reviews: List[Any]
    view_label: str

    @staticmethod
    def from_dict(obj: Any) -> 'WelcomeElement':
        assert isinstance(obj, dict)
        id = from_str(obj.get("ID"))
        data_row_id = from_str(obj.get("DataRow ID"))
        labeled_data = from_str(obj.get("Labeled Data"))
        label = Label.from_dict(obj.get("Label"))
        created_by = from_str(obj.get("Created By"))
        project_name = from_str(obj.get("Project Name"))
        created_at = from_datetime(obj.get("Created At"))
        updated_at = from_datetime(obj.get("Updated At"))
        seconds_to_label = from_float(obj.get("Seconds to Label"))
        external_id = from_str(obj.get("External ID"))
        agreement = from_int(obj.get("Agreement"))
        benchmark_agreement = from_int(obj.get("Benchmark Agreement"))
        benchmark_id = from_none(obj.get("Benchmark ID"))
        dataset_name = from_str(obj.get("Dataset Name"))
        reviews = from_list(lambda x: x, obj.get("Reviews"))
        view_label = from_str(obj.get("View Label"))
        return WelcomeElement(id, data_row_id, labeled_data, label, created_by, project_name, created_at, updated_at, seconds_to_label, external_id, agreement, benchmark_agreement, benchmark_id, dataset_name, reviews, view_label)

    def to_dict(self) -> dict:
        result: dict = {}
        result["ID"] = from_str(self.id)
        result["DataRow ID"] = from_str(self.data_row_id)
        result["Labeled Data"] = from_str(self.labeled_data)
        result["Label"] = to_class(Label, self.label)
        result["Created By"] = from_str(self.created_by)
        result["Project Name"] = from_str(self.project_name)
        result["Created At"] = self.created_at.isoformat()
        result["Updated At"] = self.updated_at.isoformat()
        result["Seconds to Label"] = to_float(self.seconds_to_label)
        result["External ID"] = from_str(self.external_id)
        result["Agreement"] = from_int(self.agreement)
        result["Benchmark Agreement"] = from_int(self.benchmark_agreement)
        result["Benchmark ID"] = from_none(self.benchmark_id)
        result["Dataset Name"] = from_str(self.dataset_name)
        result["Reviews"] = from_list(lambda x: x, self.reviews)
        result["View Label"] = from_str(self.view_label)
        return result


def welcome_from_dict(s: Any) -> List[WelcomeElement]:
    return from_list(WelcomeElement.from_dict, s)


def welcome_to_dict(x: List[WelcomeElement]) -> Any:
    return from_list(lambda x: to_class(WelcomeElement, x), x)