import os
import pickle
import time

from .Type import NLIST_AUTO, NPROBE_AUTO, GPU_CACHE_DEAULT, GPU_USE_FP16_DEFAULT
from .SearchEngine import SearchEngine, MetricType, IndexType, SearchMethod
from dpsutil.KafaWrapper import initial_producer, initial_consumer
from dpsutil.RedisWrapper import initial_redis
from dpsutil.hash import hash_now
from dpsutil.compression import compress_ndarray, decompress_ndarray, decompress
from threading import Thread


class Params(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    DEFAULT_PARAMS = {
        "method": "",
        "args": [],
        "kwargs": {},
        "topic_respond": "",
        "id": None
    }

    def __init__(self, params=None):
        if params is None:
            params = {}

        default = self.DEFAULT_PARAMS.copy()
        default.update(params)
        super().__init__(default)

    def get_dict(self):
        return dict(self)


class RequestParams(Params):
    DEFAULT_PARAMS = {
        "method": "",
        "args": [],
        "kwargs": {},
        "topic_respond": "",
        "id": None
    }


class RespondParams(Params):
    DEFAULT_PARAMS = {
        "output": None,
        "error": None,
        "id": None,
        "time_stamp": -1.
    }


def generate_unique_label():
    return f"{hash_now()}_{hash_now()}"


class SearchClient(SearchEngine):
    class SearchExecuteError(Exception):
        pass

    IndexType = IndexType
    MetricType = MetricType

    def __init__(self, receive_topic=None, group_id=None,
                 kafka_host="localhost", kafka_user_name=None, kafka_password=None,
                 redis_host="localhost:6379", redis_password="", clear_time=60):
        """
        :param receive_topic:
        :param group_id:
        :param server_host: kafka_host. Example: localhost:9092
        :param user_name: kafka_username
        :param password: kafka_password
        :param clear_time: in seconds
        """
        self.server_topic = "DPS_SEARCH_ENGINE_TESTING"
        self.sender = initial_producer(bootstrap_servers=kafka_host, sasl_plain_username=kafka_user_name,
                                       sasl_plain_password=kafka_password, value_serializer=pickle.dumps)
        self.vector_fetcher = initial_redis(redis_host, password=redis_password)

        if receive_topic is None:
            receive_topic = generate_unique_label()

        if group_id is None:
            group_id = generate_unique_label()

        self.receive_topic = receive_topic
        self.receiver = initial_consumer(receive_topic,
                                         group_id=group_id, bootstrap_servers=kafka_host,
                                         sasl_plain_username=kafka_user_name, sasl_plain_password=kafka_password,
                                         enable_auto_commit=True, value_deserializer=pickle.loads)
        self._result = {}
        self._clear_time = clear_time

        # create worker receive & clean respond.
        self._worker = [Thread(target=self._receive), Thread(target=self._clean)]

        for worker in self._worker:
            worker.start()

        if not self._check_connection():
            raise ConnectionRefusedError()

        self._data_folder = "./data/"
        if not os.path.isdir(self._data_folder):
            os.makedirs(self._data_folder)

        self._group_backup_file = "./data/group_backup.txt"

    def _check_connection(self):
        try:
            self._get_respond(self._request(SearchMethod.CHECK_CONNECTION))
            return True
        except Exception:
            return False

    def _request(self, method, *args, **kwargs):
        """
        :param method: one of methods that is supported in SearchMethod
        :param time_out: in seconds
        """
        assert method in SearchMethod, "Function isn't supported!"
        assert not self.sender._closed and not self.receiver._closed, \
            "Connection has been closed. Please create new client."

        respond_id = f"id_{hash_now()}"
        params_request = RequestParams()

        params_request.method = method.name
        params_request.args = args
        params_request.kwargs = kwargs
        params_request.topic_respond = self.receive_topic
        params_request.id = respond_id

        message = self.sender.send(self.server_topic, params_request.get_dict())

        if message.exception:
            raise message.exception

        return respond_id

    def _get_respond(self, respond_id, time_out=60):
        toc = time.time() + time_out
        while time.time() <= toc:
            if respond_id in self._result:
                respond = self._result[respond_id]
                if respond.error:
                    raise self.SearchExecuteError(respond.error)
                del self._result[respond_id]
                return respond.output
            time.sleep(1e-5)
        raise TimeoutError()

    def _clean(self):
        for idx, respond in self._result:
            if time.time() - respond['time_stamp'] > self._clear_time:
                del self._result[idx]

    def _receive(self):
        for message_block in self.receiver:
            params_respond = RespondParams(message_block.value)
            params_respond.time_stamp = time.time()
            self._result[params_respond.id] = params_respond

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

    @property
    def groups(self):
        """
        Return all groups existed.
        :return: group list
        :rtype: list
        """
        return self._get_respond(self._request(SearchMethod.GROUP_NAMES))

    def create_index(self, index_type: IndexType, metric_type: MetricType, dim: int, group_name=None):
        """
        :param index_type: support FLAT, IVF
        :param metric_type: IP, L2
        :param dim: int
        :param group_name: str
        :return: new Group's name
        :rtype: str
        """
        index_type = index_type.name
        metric_type = metric_type.name
        group_name = self._get_respond(self._request(SearchMethod.CREATE, index_type, metric_type, dim,
                                                     group_name=group_name))
        with open(self._group_backup_file, "a") as f:
            f.write(f"{group_name}\n")
        return group_name

    def get_vector(self, group_name, ids=None):
        vector_idx = self._get_respond(self._request(SearchMethod.GET, group_name, ids=ids))
        buffer = self.vector_fetcher.get(vector_idx)
        self.vector_fetcher.delete(vector_idx)
        return decompress_ndarray(buffer)

    def add_vector(self, group_name, vectors):
        """
        Add vectors into index. if ids is None, ids match with vectors that will be automatically create.
        :param group_name: name of index
        :param vectors: numpy.ndarray
        :return: ids of vectors
        """
        assert len(vectors.shape) == 2
        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)
        return self._get_respond(self._request(SearchMethod.ADD, group_name, vector_idx))

    def train(self, group_name, vectors, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO):
        """
        Train to find nlist center point that uses to speed up search but take a lot of train time.
        :param send_max_vectors:
        :param group_name: name of index
        :param vectors: numpy.ndarray
        :param nlist: int
        :param nprobe: int
        Auto change to AUTO if nprobe > nlist. When nprobe == nlist as same as brute force search.
        """
        assert len(vectors.shape) == 2
        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)
        return self._get_respond(self._request(SearchMethod.TRAIN, group_name, vector_idx, nlist=nlist, nprobe=nprobe))

    def train_add(self, group_name, vectors, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO):
        assert len(vectors.shape) == 2
        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)
        return self._get_respond(self._request(SearchMethod.TRAIN_ADD, group_name, vector_idx, nlist=nlist, nprobe=nprobe))

    def search(self, group_name, vectors, k=1):
        assert len(vectors.shape) == 2

        vector_idx = f"vector_{hash_now()}"
        self.vector_fetcher.set(vector_idx, compress_ndarray(vectors), ex=self._clear_time)

        arr_idx = self._request(SearchMethod.SEARCH, group_name, vector_idx, k=k)
        buffer = self.vector_fetcher.get(self._get_respond(arr_idx))
        self.vector_fetcher.delete(arr_idx)
        return pickle.loads(decompress(buffer))

    def remove_index(self, group_name):
        self._get_respond(self._request(SearchMethod.REMOVE_INDEX, group_name))

        with open(self._group_backup_file, "r") as f:
            groups = f.readlines()

        groups = [group for group in groups if group != group_name]
        with open(self._group_backup_file, "w") as f:
            f.writelines(groups)

    def remove_vector(self, group_name, ids):
        return self._get_respond(self._request(SearchMethod.REMOVE_VECTOR, group_name, ids))

    def retrain(self, group_name, nlist=NLIST_AUTO, nprobe=NPROBE_AUTO):
        return self._get_respond(self._request(SearchMethod.RETRAIN, group_name, nlist=nlist, nprobe=nprobe))

    def index2gpu(self, group_name, gpu_id=0, cache_size=GPU_CACHE_DEAULT, use_fp16=GPU_USE_FP16_DEFAULT):
        return self._get_respond(self._request(SearchMethod.GPU, group_name, gpu_id=gpu_id, cache_size=cache_size,
                                               use_fp16=use_fp16))

    def index2cpu(self, group_name):
        return self._get_respond(self._request(SearchMethod.CPU, group_name))

    def save(self, group_name, over_write=False):
        self._get_respond(self._request(SearchMethod.SAVE, group_name=group_name, over_write=over_write))

    def load(self, group_name):
        self._get_respond(self._request(SearchMethod.LOAD, group_name=group_name))

    def is_group_existed(self, group_name):
        return self._get_respond(self._request(SearchMethod.GROUP_EXISTED, group_name))

    def group_created(self):
        with open(self._group_backup_file, 'rt') as f:
            groups = f.read().splitlines()
        return groups
