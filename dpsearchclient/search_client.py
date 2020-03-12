import os
import pickle
import time
from threading import Thread

from dpsutil.compression import compress_ndarray, decompress
from dpsutil.hash import hash_now
from dpsutil.kafka import initial_producer, initial_consumer
from dpsutil.redis import initial_redis
from dpsutil.attrdict import DefaultDict, AttrDict

from .search_engine import SearchEngine, MetricType, IndexType, SearchMethod, GPU_CACHE_DEAULT, GPU_USE_FP16_DEFAULT, \
    NPROBE_AUTO, NLIST_AUTO, DEFAULT_TYPE


DEFAULT_REQUEST_PARAMS = {
    "method": "",
    "args": [],
    "kwargs": {},
    "topic_respond": "",
    "id": None
}

DEFAULT_RESPOND_PARAMS = {
    "output": None,
    "error": None,
    "id": None
}


def generate_unique_label():
    return f"{hash_now()}_{hash_now()}"


class SearchExecuteError(Exception):
    pass


class SearchClient(SearchEngine):
    def __init__(self, receive_topic=None, group_id=None,
                 kafka_host="localhost", kafka_user_name=None, kafka_password=None,
                 redis_host="localhost:6379", redis_password="", redis_db=0, clear_time=60):
        if group_id is None:
            group_id = generate_unique_label()

        self.server_topic = "DPS_SEARCH_ENGINE_TESTING"
        self.sender = initial_producer(bootstrap_servers=kafka_host, sasl_plain_username=kafka_user_name,
                                       sasl_plain_password=kafka_password)

        self.receiver = initial_consumer(group_id=group_id, bootstrap_servers=kafka_host,
                                         sasl_plain_username=kafka_user_name, sasl_plain_password=kafka_password,
                                         enable_auto_commit=True)

        self.vector_fetcher = initial_redis(host=redis_host, db=redis_db, password=redis_password)
        self.rp_params = DefaultDict(**DEFAULT_RESPOND_PARAMS)
        self.req_params = DefaultDict(**DEFAULT_REQUEST_PARAMS)

        if receive_topic is None:
            new_receive_topic = generate_unique_label()
        else:
            new_receive_topic = receive_topic

        while new_receive_topic in self.receiver.topics():
            new_receive_topic = f"{receive_topic}_{hash_now()}"

        self.receiver.subscribe(topics=new_receive_topic)
        self.receive_topic = new_receive_topic

        self._result = AttrDict()
        self._clear_time = clear_time

        # create worker receive & clean respond.
        self._worker = [Thread(target=self._receive), Thread(target=self._clean)]

        for worker in self._worker:
            worker.start()

        self._check_connection()

        self._data_folder = "./data/"
        if not os.path.isdir(self._data_folder):
            os.makedirs(self._data_folder)

    def _check_connection(self):
        self._get_respond(self._request(SearchMethod.CHECK_CONNECTION))

    def _request(self, method, *args, **kwargs):
        """
        :param method: one of methods that is supported in SearchMethod
        :param time_out: in seconds
        """
        assert method in SearchMethod, "Function isn't supported!"
        assert not self.sender._closed and not self.receiver._closed, \
            "Connection has been closed. Please create new client."

        respond_id = f"id_{hash_now()}"

        self.req_params.clear()
        self.req_params.method = method.name
        self.req_params.args = args
        self.req_params.kwargs = kwargs
        self.req_params.topic_respond = self.receive_topic
        self.req_params.id = respond_id

        message = self.sender.send(self.server_topic, self.req_params.to_buffer())

        if message.exception:
            raise message.exception

        return respond_id

    def _get_respond(self, respond_id, time_out=60):
        toc = time.time() + time_out
        while time.time() <= toc:
            if respond_id in self._result:
                respond = self._result[respond_id]
                if respond.error:
                    raise SearchExecuteError(respond.error)
                del self._result[respond_id]
                return respond.output
            time.sleep(1e-5)
        raise TimeoutError()

    def _clean(self):
        while 1:
            for idx, respond in self._result.items():
                if time.time() - respond['time_stamp'] > self._clear_time:
                    del self._result[idx]
            time.sleep(5)

    def _receive(self):
        for message_block in self.receiver:

            self.rp_params.clear()
            self.rp_params.from_buffer(message_block.value)

            rp_params = self.rp_params.copy()
            rp_params.time_stamp = time.time()

            self._result[rp_params.id] = rp_params

    def close(self, time_out=30):
        self.sender.close()
        self.receiver.close()

        toc = time.time() + time_out
        while time.time() < toc:
            if self.sender._closed and self.receiver._closed:
                break
            time.sleep(0.5)

        for worker in self._worker:
            worker._stop()

    def __del__(self):
        self.close()

    def __exit__(self):
        self.close()

    def create(self, group_name, index_type: IndexType, metric_type: MetricType, dim: int, dtype=DEFAULT_TYPE,
               with_labels=False):
        index_type = index_type.name
        metric_type = metric_type.name
        is_success = self._get_respond(self._request(SearchMethod.CREATE, group_name, index_type, metric_type, dim,
                                                     dtype=dtype, with_labels=with_labels))
        return is_success

    def get(self, group_name, ids=None):
        vector_idx = self._get_respond(self._request(SearchMethod.GET, group_name, ids=ids))
        buffer = self.vector_fetcher.get(vector_idx)
        self.vector_fetcher.delete(vector_idx)
        return pickle.loads(decompress(buffer))

    def add(self, group_name, vectors, labels=None, filter_unique=False, filter_distance=1e-6):
        assert len(vectors.shape) == 2
        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)
        return self._get_respond(self._request(SearchMethod.ADD, group_name, vector_idx, labels=labels,
                                               filter_unique=filter_unique, filter_distance=filter_distance))

    def search(self, group_name, vectors, k=1):
        assert len(vectors.shape) == 2

        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)

        arr_idx = self._request(SearchMethod.SEARCH, group_name, vector_idx, k=k)
        buffer = self.vector_fetcher.get(self._get_respond(arr_idx))
        self.vector_fetcher.delete(arr_idx)
        return pickle.loads(decompress(buffer))

    def remove(self, group_name):
        self._get_respond(self._request(SearchMethod.REMOVE_INDEX, group_name))

    def remove_vector(self, group_name, ids):
        return self._get_respond(self._request(SearchMethod.REMOVE_VECTOR, group_name, ids))

    def train(self, group_name, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO, filter_unique=False, filter_distance=1e-6,
              gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        return self._get_respond(self._request(SearchMethod.TRAIN, group_name, nlist=nlist, nprobe=nprobe,
                                               filter_unique=filter_unique, filter_distance=filter_distance,
                                               gpu_id=gpu_id, cache_size=cache_size, use_fp16=use_fp16))

    def index2gpu(self, group_name, gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        return self._get_respond(self._request(SearchMethod.GPU, group_name, gpu_id=gpu_id, cache_size=cache_size,
                                               use_fp16=use_fp16))

    def index2cpu(self, group_name):
        return self._get_respond(self._request(SearchMethod.CPU, group_name))

    def save(self, group_name, over_write=False):
        return bool(self._get_respond(self._request(SearchMethod.SAVE, group_name=group_name, over_write=over_write)))

    def load(self, group_name, with_labels=False):
        try:
            self._get_respond(self._request(SearchMethod.LOAD, group_name=group_name, with_labels=with_labels))
        except SearchExecuteError:
            return False

    def is_existed(self, group_name):
        return self._get_respond(self._request(SearchMethod.GROUP_EXISTED, group_name))

    def count_label(self, group_name, label):
        return self._get_respond(self._request(SearchMethod.COUNT_LABEL, group_name, label))


__all__ = ['SearchExecuteError', 'SearchClient']
