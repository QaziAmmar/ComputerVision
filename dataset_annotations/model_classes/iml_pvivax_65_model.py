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

# this is testing file for parsing of json data
import json
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union, TypeVar, Callable, Type, cast
from datetime import datetime
import dateutil.parser


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_str(x: Any) -> str:
    assert isinstance(x, str)
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


def from_none(x: Any) -> Any:
    assert x is None
    return x


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


class CreatedBy(Enum):
    MSCS18006_ITU_EDU_PK = "mscs18006@itu.edu.pk"
    MSCS18031_ITU_EDU_PK = "mscs18031@itu.edu.pk"


class DatasetName(Enum):
    PIVIAX_IML_65 = "Piviax_IML_65"


@dataclass
class Geometry:
    x: int
    y: int

    @staticmethod
    def from_dict(obj: Any) -> 'Geometry':
        assert isinstance(obj, dict)
        x = from_int(obj.get("x"))
        y = from_int(obj.get("y"))
        return Geometry(x, y)

    def to_dict(self) -> dict:
        result: dict = {}
        result["x"] = from_int(self.x)
        result["y"] = from_int(self.y)
        return result


@dataclass
class RedBloodCell:
    geometry: List[Geometry]

    @staticmethod
    def from_dict(obj: Any) -> 'RedBloodCell':
        assert isinstance(obj, dict)
        geometry = from_list(Geometry.from_dict, obj.get("geometry"))
        return RedBloodCell(geometry)

    def to_dict(self) -> dict:
        result: dict = {}
        result["geometry"] = from_list(lambda x: to_class(Geometry, x), self.geometry)
        return result


@dataclass
class LabelClass:
    red_blood_cell: List[RedBloodCell]

    @staticmethod
    def from_dict(obj: Any) -> 'LabelClass':
        assert isinstance(obj, dict)
        red_blood_cell = from_list(RedBloodCell.from_dict, obj.get("Red Blood Cell"))
        return LabelClass(red_blood_cell)

    def to_dict(self) -> dict:
        result: dict = {}
        result["Red Blood Cell"] = from_list(lambda x: to_class(RedBloodCell, x), self.red_blood_cell)
        return result


class ProjectName(Enum):
    PLASMODIUM_VIVAX_IML_65 = "plasmodium_vivax_IML_65"


@dataclass
class WelcomeElement:
    id: str
    data_row_id: str
    labeled_data: str
    label: Union[LabelClass, str]
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

    @staticmethod
    def from_dict(obj: Any) -> 'WelcomeElement':
        assert isinstance(obj, dict)
        id = from_str(obj.get("ID"))
        data_row_id = from_str(obj.get("DataRow ID"))
        labeled_data = from_str(obj.get("Labeled Data"))
        label = from_union([LabelClass.from_dict, from_str], obj.get("Label"))
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
        result["Label"] = from_union([lambda x: to_class(LabelClass, x), from_str], self.label)
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

