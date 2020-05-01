import numpy

from .meta_data import IndexType, MetricType

NPROBE_AUTO = 0
NLIST_AUTO = 0

DEFAULT_TYPE = numpy.float32

GPU_USE_FP16_DEFAULT = True
GPU_CACHE_DEAULT = 256 * 1024 * 1024


class SearchEngineMethods(object):
    def create(self, group_name, index_type: IndexType, metric_type: MetricType, dim: int, dtype=DEFAULT_TYPE,
               with_labels=False):
        """
        Create new index with name and metric's type.
        Optionals:
        - nlist: if you want to define first, this params can edit after on train step. (Only IVF type)
        - index_size: This define size of index in memory or vram if working on GPU.
        :return: group_name
        """
        pass

    def add(self, group_name, vectors, labels=None, filter_unique=False, filter_distance=1e-6):
        """
        Add vectors into index.
        """
        pass

    def get(self, group_name, ids=None):
        pass

    def search(self, group_name, vectors, k=1):
        """
        :return: ids of vectors
        """
        pass

    def save(self, group_name, over_write=False):
        pass

    def load(self, group_name, with_labels=False):
        pass

    def remove(self, group_name):
        pass

    def remove_vector(self, group_name, ids):
        pass

    def train(self, group_name, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO, filter_unique=False, filter_distance=1e-6,
              gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        pass

    def index2gpu(self, group_name, gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        pass

    def index2cpu(self, group_name):
        pass

    def is_existed(self, group_name):
        pass

    def count_label(self, group_name, label):
        pass
