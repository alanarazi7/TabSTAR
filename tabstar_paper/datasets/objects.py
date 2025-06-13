from enum import StrEnum


class SupervisedTask(StrEnum):
    # openml.tasks.TaskType doesn't separate binary and multiclass classification, so we redefine it
    REGRESSION = "📈 regression"
    BINARY = "⚖️ binary"
    MULTICLASS = "🎨 multiclass"


class FeatureType(StrEnum):
    CATEGORICAL = "🏷️ categorical"
    NUMERIC = "🔢 numeric"
    TEXT = "📝 text"
    DATE = "📅 date"
    BOOLEAN = "☑️ boolean"
