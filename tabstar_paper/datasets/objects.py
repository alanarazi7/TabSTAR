from enum import Enum


class SupervisedTask(Enum):
    # openml.tasks.TaskType doesn't separate binary and multiclass classification, so we redefine it
    REGRESSION = "📈 regression"
    BINARY = "⚖️ binary"
    MULTICLASS = "🎨 multiclass"


class FeatureType(Enum):
    CATEGORICAL = "🏷️ categorical"
    NUMERIC = "🔢 numeric"
    TEXT = "📝 text"
    DATE = "📅 date"
    BOOLEAN = "☑️ boolean"
