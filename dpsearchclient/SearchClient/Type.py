from enum import Enum, unique

METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1

NPROBE_AUTO = 0
NLIST_AUTO = 0

GPU_USE_FP16_DEFAULT = True
GPU_CACHE_DEAULT = 256 * 1024 * 1024


@unique
class IndexType(Enum):
    FLAT = 0
    IVF = 1


@unique
class MetricType(Enum):
    INNER_PRODUCT = METRIC_INNER_PRODUCT
    L2 = METRIC_L2


@unique
class DistanceType(Enum):
    CosineDistance = 0
    CosineSimilarity = 1
