import sys
import hashlib
import json
from bson import ObjectId
import numpy
import pickle


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, ObjectId):
            return str(obj)
        # Note, the following mapping is bad :( but usefull for us
        elif isinstance(obj, type):
            return str(obj)
        else:
            return super(JsonEncoder, self).default(obj)


def get_hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, cls=JsonEncoder).encode('utf-8')).hexdigest()


def ipython_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'notebook'
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip


def get_tqdm():
    from tqdm import tqdm
    return tqdm
    ip = ipython_info()
    if ip == "terminal" or not ip:
        from tqdm import tqdm
        return tqdm
    else:
        try:
            from tqdm import tqdm_notebook
            return tqdm_notebook
        except ImportError:
            from tqdm import tqdm
            return tqdm


def if_none(first, second):
    return second if first is None else first


def save_pkl(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, "rb") as file:
        res = pickle.load(file)
    return res
