from enum import Enum, unique


@unique
class IndexType(Enum):
    FLAT = 0
    IVF = 1


@unique
class MetricType(Enum):
    INNER_PRODUCT = 0
    L2 = 1


@unique
class DistanceType(Enum):
    CosineDistance = 0
    CosineSimilarity = 1


__all__ = ['IndexType', 'MetricType', 'DistanceType']
