import os
import pickle
import time
from queue import Queue
from threading import Thread

from dpsutil.attrdict import DefaultDict, AttrDict
from dpsutil.compression import compress_ndarray, decompress
from dpsutil.hash import hash_now
from dpsutil.kafka import initial_producer, initial_consumer
from dpsutil.redis import initial_redis

from .search_engine import SearchEngineMethods, \
    MetricType, IndexType, GPU_CACHE_DEAULT, GPU_USE_FP16_DEFAULT, \
    NPROBE_AUTO, NLIST_AUTO, DEFAULT_TYPE


def generate_unique_label():
    return hash_now()


class SearchExecuteError(Exception):
    pass


class SearchClient(SearchEngineMethods):
    """Contact with SearchEngineServer via Kafka, Redis"""

    def __init__(self,
                 receive_topic=None,
                 group_id=None,
                 kafka_host="localhost",
                 kafka_username=None,
                 kafka_password=None,
                 redis_host="localhost:6379",
                 redis_password="",
                 redis_db=0,
                 clear_time=60,
                 session_file="./session_bk.txt"):

        if group_id is None:
            group_id = generate_unique_label()

        self.server_topic = "DPS_SEARCH_ENGINE_TESTING"
        self.sender = initial_producer(bootstrap_servers=kafka_host,
                                       sasl_plain_username=kafka_username,
                                       sasl_plain_password=kafka_password)

        self.receiver = initial_consumer(bootstrap_servers=kafka_host,
                                         group_id=group_id,
                                         sasl_plain_username=kafka_username,
                                         sasl_plain_password=kafka_password,
                                         enable_auto_commit=True)

        self.vector_fetcher = initial_redis(host=redis_host, db=redis_db, password=redis_password)

        self._req_params = DefaultDict(method="", session=None, group_name=None,
                                       args=[], kwargs={}, topic_respond="", id=None)
        self._res_params = DefaultDict(output=None, error=None, id=None)

        if receive_topic is None:
            new_receive_topic = generate_unique_label()
        else:
            new_receive_topic = receive_topic

        while new_receive_topic in self.receiver.topics():
            new_receive_topic = f"{receive_topic}{hash_now()}"

        self.receiver.subscribe(topics=new_receive_topic)
        self.receive_topic = new_receive_topic
        self.session_file = session_file

        self._result = Queue()
        self._clear_time = clear_time
        self._stopped = False

        # create worker which receive respond.
        Thread(target=self._receive).start()
        self.check_connection()

    def _request(self, method, *args, **kwargs):
        """
        :param method: one of methods that is supported in SearchMethod
        :param time_out: in seconds
        """
        self._req_params.clear()
        self._req_params.method = method

        if len(args) > 0:
            self._req_params.group_name = args[0]
            self._req_params.args = args[1:]

        self._req_params.kwargs = kwargs
        self._req_params.topic_respond = self.receive_topic
        self._req_params.id = f"id_{generate_unique_label()}"

        message = self.sender.send(self.server_topic, self._req_params.to_buffer(compress_type=0))

        if message.exception:
            raise message.exception

        while 1:
            respond = self._result.get()
            if respond.id == self._req_params.id:
                break

        if respond.error:
            raise SearchExecuteError(respond.error)

        return respond.output

    def _receive(self):
        for message_block in self.receiver:
            if self._stopped:
                break

            self._res_params.clear()
            self._res_params(message_block.value)
            _res_params = AttrDict(self._res_params)
            _res_params.time_stamp = time.time()
            self._result.put(_res_params)

    def close(self):
        self.sender.close()
        self.receiver.close()

    def __del__(self):
        self.close()

    def __exit__(self):
        self.close()

    def check_connection(self):
        if self.session_file and os.path.isfile(self.session_file):
            with open(self.session_file, 'r') as f:
                session = f.read(32)
                self._req_params.setdefault('session', session)
                self._req_params.clear('session')

        session = self._request(self.check_connection.__name__)

        if not session:
            return

        os.rename(self.session_file, f"{self.session_file}_backup_{time.time()}")
        with open(self.session_file, "w") as f:
            f.write(session)

        self._req_params.setdefault('session', session)
        self._req_params.clear('session')

    def create(self, group_name, index_type: IndexType, metric_type: MetricType, dim: int, dtype=DEFAULT_TYPE,
               with_labels=False):
        index_type = index_type.name
        metric_type = metric_type.name
        is_success = self._request(self.create.__name__, group_name, index_type, metric_type, dim,
                                   dtype=dtype, with_labels=with_labels)
        return is_success

    def get(self, group_name, ids=None):
        vector_idx = self._request(self.get.__name__, group_name, ids=ids)
        buffer = self.vector_fetcher.get(vector_idx)
        self.vector_fetcher.delete(vector_idx)
        return pickle.loads(decompress(buffer))

    def add(self, group_name, vectors, labels=None, filter_unique=False, filter_distance=1e-6):
        assert len(vectors.shape) == 2
        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)
        return self._request(self.add.__name__, group_name, vector_idx, labels=labels,
                             filter_unique=filter_unique, filter_distance=filter_distance)

    def search(self, group_name, vectors, k=1):
        assert len(vectors.shape) == 2

        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)

        arr_idx = self._request(self.search.__name__, group_name, vector_idx, k=k)
        buffer = self.vector_fetcher.get(arr_idx)
        self.vector_fetcher.delete(arr_idx)
        return pickle.loads(decompress(buffer))

    def remove(self, group_name):
        self._request(self.remove.__name__, group_name)

    def remove_vector(self, group_name, ids):
        return self._request(self.remove_vector.__name__, group_name, ids)

    def train(self, group_name, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO, filter_unique=False, filter_distance=1e-6,
              gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        return self._request(self.train.__name__, group_name, nlist=nlist, nprobe=nprobe,
                             filter_unique=filter_unique, filter_distance=filter_distance,
                             gpu_id=gpu_id, cache_size=cache_size, use_fp16=use_fp16)

    def index2gpu(self, group_name, gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        return self._request(self.index2gpu.__name__, group_name, gpu_id=gpu_id, cache_size=cache_size,
                             use_fp16=use_fp16)

    def index2cpu(self, group_name):
        return self._request(self.index2cpu.__name__, group_name)

    def save(self, group_name, over_write=False):
        return bool(self._request(self.save.__name__, group_name, over_write=over_write))

    def load(self, group_name, with_labels=False):
        try:
            self._request(self.load.__name__, group_name, with_labels=with_labels)
        except SearchExecuteError:
            return False

    def is_existed(self, group_name):
        return self._request(self.is_existed.__name__, group_name)

    def count_label(self, group_name, label):
        return self._request(self.count_label.__name__, group_name, label)


__all__ = ['SearchExecuteError', 'SearchClient']
