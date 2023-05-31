import pickle
import time
from functools import wraps
from multiprocessing import Queue

import dill
import numpy as np
from tqdm import tqdm
from os.path import exists
# pylint: disable = missing-function-docstring


def exec_time(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.process_time()
        func(*args, **kwargs)
        end_time = time.process_time()
        total_time = (end_time - start_time)
        return total_time

    return timeit_wrapper


def append_tensor(old_stack, new_tensor):
    """
    Appends two tensors on a new axis. If `old_stack` shape is the same of `new_tensor`
    they are stacked together with `np.stack`. Otherwise if `old_stack` shape is bigger
    than `new_tensor` shape's, `new_tensor` is expanded on axis 0 and they are concatenated.
   
    Args:
        old_stack: old stack, it's shape must be >= new_tensor's shape
        new_tensor: new tensor that will be appended to `old_stack`
    """
    if len(old_stack.shape) == len(new_tensor.shape):
        return np.stack((old_stack, new_tensor))
    else:
        nt = np.reshape(new_tensor, (1, *new_tensor.shape))
        return np.concatenate((old_stack, nt))


def generate_data(data=None, batch_size=6000):
    """
    Generates a "complex" data structure like a batch of
    data from "The AI-Economist" reinforcement learning
    environment.

    Pretty much hard coded, it can be made clearer and simpler
    with a dictionary -> `key`:`shape tuple`
    """
    flag = 0
    if data == None:
        flag = 1
        data = {
            '0': {
                'obs': {}
            },
            '1': {
                'obs': {}
            },
            '2': {
                'obs': {}
            },
            '3': {
                'obs': {}
            },
            'p': {
                'obs': {}
            },
        }
        for key in data.keys():
            if mapping_function(key):
                # True -> agent
                data[key]['obs']['world-map'] = np.random.rand(7, 11, 11)
                data[key]['obs']['world-map_idx'] = np.random.rand(2, 11, 11)
                data[key]['obs']['time'] = np.random.rand(1)
                data[key]['obs']['flat'] = np.random.rand(136)
                data[key]['obs']['action_mask'] = np.random.rand(50)

                data[key]['action'] = np.random.rand(1)
                data[key]['log_action'] = np.random.rand(1)
                data[key]['reward'] = np.random.rand(1)
                data[key]['is_terminal'] = np.random.rand(1)
            else:
                # False -> planner
                data[key]['obs']['world-map'] = np.random.rand(6, 25, 25)
                data[key]['obs']['world-map_idx'] = np.random.rand(2, 25, 25)
                data[key]['obs']['time'] = np.random.rand(1)
                data[key]['obs']['flat'] = np.random.rand(86)
                data[key]['obs']['action_mask'] = np.random.rand(154)
                data[key]['obs']['p0'] = np.random.rand(8)
                data[key]['obs']['p1'] = np.random.rand(8)
                data[key]['obs']['p2'] = np.random.rand(8)
                data[key]['obs']['p3'] = np.random.rand(8)

                data[key]['action'] = np.random.rand(1)
                data[key]['log_action'] = np.random.rand(1)
                data[key]['reward'] = np.random.rand(1)
                data[key]['is_terminal'] = np.random.rand(1)

    # Generate data of 4 agents + 1 planner
    for _ in range(batch_size - flag):
        for key in data.keys():
            if mapping_function(key):
                # True -> agent
                data[key]['obs']['world-map'] = append_tensor(
                    data[key]['obs']['world-map'], np.random.rand(7, 11, 11))
                data[key]['obs']['world-map_idx'] = append_tensor(
                    data[key]['obs']['world-map_idx'],
                    np.random.rand(2, 11, 11))
                data[key]['obs']['time'] = append_tensor(
                    data[key]['obs']['time'], np.random.rand(1))
                data[key]['obs']['flat'] = append_tensor(
                    data[key]['obs']['flat'], np.random.rand(136))
                data[key]['obs']['action_mask'] = append_tensor(
                    data[key]['obs']['action_mask'], np.random.rand(50))

                data[key]['action'] = append_tensor(data[key]['action'],
                                                    np.random.rand(1))
                data[key]['log_action'] = append_tensor(
                    data[key]['log_action'], np.random.rand(1))
                data[key]['reward'] = append_tensor(data[key]['reward'],
                                                    np.random.rand(1))
                data[key]['is_terminal'] = append_tensor(
                    data[key]['is_terminal'], np.random.rand(1))
            else:
                # False -> planner
                data[key]['obs']['world-map'] = append_tensor(
                    data[key]['obs']['world-map'], np.random.rand(6, 25, 25))
                data[key]['obs']['world-map_idx'] = append_tensor(
                    data[key]['obs']['world-map_idx'],
                    np.random.rand(2, 25, 25))
                data[key]['obs']['time'] = append_tensor(
                    data[key]['obs']['time'], np.random.rand(1))
                data[key]['obs']['flat'] = append_tensor(
                    data[key]['obs']['flat'], np.random.rand(86))
                data[key]['obs']['action_mask'] = append_tensor(
                    data[key]['obs']['action_mask'], np.random.rand(154))
                data[key]['obs']['p0'] = append_tensor(data[key]['obs']['p0'],
                                                       np.random.rand(8))
                data[key]['obs']['p1'] = append_tensor(data[key]['obs']['p1'],
                                                       np.random.rand(8))
                data[key]['obs']['p2'] = append_tensor(data[key]['obs']['p2'],
                                                       np.random.rand(8))
                data[key]['obs']['p3'] = append_tensor(data[key]['obs']['p3'],
                                                       np.random.rand(8))

                data[key]['action'] = append_tensor(data[key]['action'],
                                                    np.random.rand(1))
                data[key]['log_action'] = append_tensor(
                    data[key]['log_action'], np.random.rand(1))
                data[key]['reward'] = append_tensor(data[key]['reward'],
                                                    np.random.rand(1))
                data[key]['is_terminal'] = append_tensor(
                    data[key]['is_terminal'], np.random.rand(1))
    return data


def mapping_function(key):
    if str(key).isdigit() or key == "a":
        return True
    return False


@exec_time
def pickle_unsafe_disk(data):
    data_file = open(f"/tmp/test.bin", "wb")
    pickle.dump(data, data_file)
    data_file.close()


@exec_time
def pickle_unsafe_ram(data):
    data_file = open(f"/dev/shm/test.bin", "wb")
    pickle.dump(data, data_file)
    data_file.close()


@exec_time
def pickle_unsafe_ram_hp(data):
    data_file = open(f"/dev/shm/test.bin", "wb")
    pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)
    data_file.close()


@exec_time
def dill_unsafe_disk(data):
    data_file = open(f"/tmp/test.bin", "wb")
    dill.dump(data, data_file)
    data_file.close()


@exec_time
def dill_unsafe_ram(data):
    data_file = open(f"/dev/shm/test.bin", "wb")
    dill.dump(data, data_file)
    data_file.close()

@exec_time
def dill_safe_ram(data):
    with open(f"/dev/shm/test.bin", "wb") as data_file:
        dill.dump(data, data_file)

@exec_time
def dill_safe_ram_hp(data):
    with open(f"/dev/shm/test.bin", "wb") as data_file:
        dill.dump(data, data_file, dill.HIGHEST_PROTOCOL)

@exec_time
def pickle_safe_disk(data):
    with open(f"/tmp/test.bin", "wb") as data_file:
        pickle.dump(data, data_file)


@exec_time
def pickle_safe_ram(data):
    with open(f"/dev/shm/test.bin", "wb") as data_file:
        pickle.dump(data, data_file)


@exec_time
def pickle_safe_ram_hp(data):
    with open(f"/dev/shm/test.bin", "wb") as data_file:
        pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)


@exec_time
def send_queue(data):
    queue.put(data)


@exec_time
def unpickle_unsafe_disk(a):
    data_file = open(f"/tmp/test.bin", "rb")
    data = pickle.load(data_file)
    data_file.close()


@exec_time
def unpickle_unsafe_ram(a):
    data_file = open(f"/dev/shm/test.bin", "rb")
    data = pickle.load(data_file)
    data_file.close()


@exec_time
def undill_unsafe_disk(a):
    data_file = open(f"/tmp/test.bin", "rb")
    data = dill.load(data_file)
    data_file.close()


@exec_time
def undill_unsafe_ram(a):
    data_file = open(f"/dev/shm/test.bin", "rb")
    data = dill.load(data_file)
    data_file.close()

@exec_time
def undill_safe_ram(a):
    with open(f"/dev/shm/test.bin", "rb") as data_file:
        data = dill.load(data_file)

@exec_time
def unpickle_safe_disk(a):
    with open(f"/tmp/test.bin", "rb") as data_file:
        data = pickle.load(data_file)


@exec_time
def unpickle_safe_ram(a):
    with open(f"/dev/shm/test.bin", "rb") as data_file:
        data = pickle.load(data_file)


@exec_time
def get_queue(a):
    data = queue.get()


def log_time(name: str, time_log: list):
    path = f"logs/{name}.csv"
    if not exists(path):
        with open(path, "a+") as log:
            comma = ""
            for _ in range(REPETITIONS):
                comma += ","

            log.write(comma)
            log.write("\n")
            log.write(f"{','.join(map(str, time_log))}\n")
    else:
        with open(path, "a+") as log:
            log.write(f"{','.join(map(str, time_log))}\n")


if __name__ == '__main__':
    # Parameters:
    BATCH_SIZE = 500
    REPETITIONS = 100

    STEP_SIZE = 500
    BIG_SIZE = 6000

    # Queue
    queue = Queue()

    # Generate data
    iterations = [1, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]#, 650, 750, 1000, 1500, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

    communication_methods = {
        # "pickle_unsafe_disk": pickle_unsafe_disk,
        # "unpickle_unsafe_disk": unpickle_unsafe_disk,
        # "pickle_unsafe_ram": pickle_unsafe_ram,
        # "unpickle_unsafe_ram": unpickle_unsafe_ram,
        # "pickle_unsafe_ram_hp": pickle_unsafe_ram_hp,


        # "pickle_safe_disk": pickle_safe_disk,
        # "unpickle_safe_disk": unpickle_safe_disk,
        "pickle_safe_ram": pickle_safe_ram,
        "unpickle_safe_ram": unpickle_safe_ram,
        "pickle_safe_ram_hp": pickle_safe_ram_hp,
        "unpickle_safe_ram_hp": unpickle_safe_ram,

        # "dill_unsafe_disk": dill_unsafe_disk,
        # "undill_unsafe_disk": undill_unsafe_disk,
        # "dill_unsafe_ram": dill_unsafe_ram,
        # "undill_unsafe_ram": undill_unsafe_ram,

        "dill_safe_ram":dill_safe_ram,
        "undill_safe_ram": undill_safe_ram,        
        "dill_safe_ram_hp":dill_safe_ram_hp,
        "undill_safe_ram_hp": undill_safe_ram,
        
        # "send_queue": send_queue,
        # "get_queue": get_queue,
    }
    data = None
    for index in tqdm(range(len(iterations))):
        data = generate_data(data=data,
                             batch_size=iterations[index] - iterations[index - 1])
        print(data['0']['obs']['world-map'].shape)

        for key, value in communication_methods.items():
            time_log = []
            for _ in range(REPETITIONS):
                time_log.append(value(data))

            log_time(key, time_log)
