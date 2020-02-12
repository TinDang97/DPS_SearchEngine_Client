from enum import unique, Enum

from .MetaData import IndexType, MetricType, NLIST_AUTO, NPROBE_AUTO, GPU_USE_FP16_DEFAULT, GPU_CACHE_DEAULT, \
    DEFAULT_TYPE


@unique
class SearchMethod(Enum):
    LOAD = "LOAD"
    SAVE = "SAVE"
    SEARCH = "SEARCH"
    CREATE = "CREATE"
    TRAIN = "TRAIN"
    TRAIN_ADD = "TRAIN_ADD"
    RETRAIN = "RETRAIN"
    ADD = "ADD"
    COUNT_LABEL = "COUNT_LABEL"
    GET = "GET"
    REMOVE_INDEX = "REMOVE_INDEX"
    REMOVE_VECTOR = "REMOVE_VECTOR"
    GPU = "GPU"
    CPU = "CPU"
    GROUP_EXISTED = "GROUP_EXISTED"
    CHECK_CONNECTION = "CHECK_CONNECTION"


class SearchEngine(object):
    def create_group(self, group_name, index_type: IndexType, metric_type: MetricType, dim: int, dtype=DEFAULT_TYPE,
                     with_labels=False):
        """
        Create new index with name and metric's type.
        Optionals:
        - nlist: if you want to define first, this params can edit after on train step. (Only IVF type)
        - index_size: This define size of index in memory or vram if working on GPU.
        :param with_labels:
        :param dtype:
        :param group_name:
        :param index_type: support "FLAT" or "IVF"
        :param metric_type: support "L2" or "IP"
        :param dim: dim
        :param use_gpu: True or False
        :param index_size: Size of cache memory that uses in GPU if use_gpu=true
        :return: group_name
        """
        pass

    def index_cpu_to_gpus(self, group_name):
        """
        :param group_name:
        :return: bool
        """
        pass

    def add_vector(self, group_name, vectors, labels=None, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO, filter_unique=False, filter_distance=1e-6):
        """
        Add vectors into index.
        :param labels:
        :param filter_distance:
        :param filter_unique:
        :param nprobe:
        :param nlist:
        :param group_name: name of index
        :param vectors: numpy.ndarray
        :return: ids of vectors
        """
        pass

    def get_vector(self, group_name, ids=None):
        pass

    def search(self, group_name, vectors, k=1):
        """
        :param k: number of neighbour
        :param group_name: name of index
        :param vectors: numpy.ndarray
        :return: ids of vectors
        """
        pass

    def save(self, group_name, over_write=False):
        pass

    def load(self, group_name):
        pass

    def remove_index(self, group_name):
        pass

    def remove_vector(self, group_name, ids):
        pass

    def retrain(self, group_name, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO, filter_unique=False, filter_distance=1e-6,
                gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        pass

    def index2gpu(self, group_name, gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        pass

    def index2cpu(self, group_name):
        pass

    def is_group_existed(self, group_name):
        pass

    def count_label(self, group_name, label):
        pass
