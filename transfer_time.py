from functools import wraps
import time
import dill
import pickle
import numpy as np
from multiprocessing import Queue
from tqdm import tqdm


def exec_time(func):  # pylint: disable = missing-function-docstring

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.process_time()  #perf_counter()
        result = func(*args, **kwargs)
        end_time = time.process_time()  #perf_counter()
        total_time = (end_time - start_time) / REPETITIONS
        # print(f"Function {func.__name__} Took {total_time} seconds")
        return total_time, result

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

def generate_data(batch_size=6000):
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
        if policy_mapping_function(key):
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
    for _ in range(batch_size - 1):
        for key in data.keys():
            if policy_mapping_function(key):
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

def add_data(data, batch_size):
    for _ in range(batch_size):
        for key in data.keys():
            if policy_mapping_function(key):
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

def policy_mapping_function(key):
    if str(key).isdigit() or key == "a":
        return True
    return False

@exec_time
def pickle_unsafe_disk(data):
    for _ in range(REPETITIONS):
        data_file = open(f"/tmp/test.bin", "wb")
        pickle.dump(data, data_file)
        data_file.close()

@exec_time
def pickle_unsafe_ram(data):
    for _ in range(REPETITIONS):
        data_file = open(f"/dev/shm/test.bin", "wb")
        pickle.dump(data, data_file)
        data_file.close()

@exec_time
def pickle_unsafe_ram_hp(data):
    for _ in range(REPETITIONS):
        data_file = open(f"/dev/shm/test.bin", "wb")
        pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)
        data_file.close()

@exec_time
def dill_unsafe_disk(data):
    for _ in range(REPETITIONS):
        data_file = open(f"/tmp/test.bin", "wb")
        dill.dump(data, data_file)
        data_file.close()

@exec_time
def dill_unsafe_ram(data):
    for _ in range(REPETITIONS):
        data_file = open(f"/dev/shm/test.bin", "wb")
        pickle.dump(data, data_file)
        data_file.close()


@exec_time
def pickle_safe_disk(data):
    for n in range(REPETITIONS):
        with open(f"/tmp/test.bin", "wb") as data_file:
            pickle.dump(data, data_file)


@exec_time
def pickle_safe_ram(data):
    for n in range(REPETITIONS):
        with open(f"/dev/shm/test.bin", "wb") as data_file:
            pickle.dump(data, data_file)


@exec_time
def pickle_safe_ram_hp(data):
    for n in range(REPETITIONS):
        with open(f"/dev/shm/test.bin", "wb") as data_file:
            pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)


@exec_time
def send_queue(data):
    for _ in range(REPETITIONS):
        queue.put(data)


@exec_time
def unpickle_unsafe_disk():
    for _ in range(REPETITIONS):
        data_file = open(f"/tmp/test.bin", "rb")
        data = pickle.load(data_file)
        data_file.close()


@exec_time
def unpickle_unsafe_ram():
    for _ in range(REPETITIONS):
        data_file = open(f"/dev/shm/test.bin", "rb")
        data = pickle.load(data_file)
        data_file.close()


@exec_time
def undill_unsafe_disk():
    for _ in range(REPETITIONS):
        data_file = open(f"/tmp/test.bin", "rb")
        data = dill.load(data_file)
        data_file.close()


@exec_time
def undill_unsafe_ram():
    for _ in range(REPETITIONS):
        data_file = open(f"/dev/shm/test.bin", "rb")
        data = dill.load(data_file)
        data_file.close()


@exec_time
def unpickle_safe_disk():
    for _ in range(REPETITIONS):
        with open(f"/tmp/test.bin", "rb") as data_file:
            data = pickle.load(data_file)


@exec_time
def unpickle_safe_ram():
    for _ in range(REPETITIONS):
        with open(f"/dev/shm/test.bin", "rb") as data_file:
            data = pickle.load(data_file)


@exec_time
def get_queue():
    for _ in range(REPETITIONS):
        data = queue.get()


# TODO: logger, repeater, graphs

if __name__ == '__main__':
    # Parameters:
    BATCH_SIZE = 500
    REPETITIONS = 100

    STEP_SIZE = 500
    BIG_SIZE = 6000

    # Queue
    queue = Queue()

    # Generate data
    data = generate_data(BATCH_SIZE)
    iterations = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]

    for index in tqdm(range(len(iterations))):
        if index != 0:
           data = add_data(data,
                           batch_size=iterations[index] -
                           iterations[index - 1])
        
        # Pickle unsafe
        # pud, _ = pickle_unsafe_disk(data)
        # uud, _ = unpickle_unsafe_disk()

        # pur, _ = pickle_unsafe_ram(data)
        # uur, _ = unpickle_unsafe_ram()

        # purh, _ = pickle_unsafe_ram_hp(data)
        # uur1, _ = unpickle_unsafe_ram()

        # dud, _ = dill_unsafe_disk(data)
        # uud1, _ = undill_unsafe_disk()

        # dur, _ = dill_unsafe_ram(data)
        # uur2, _ = undill_unsafe_ram()

        psd, _ = pickle_safe_disk(data)
        usd, _ = unpickle_safe_disk()

        psr, _ = pickle_safe_ram(data)
        usr, _ = unpickle_safe_ram()

        psrh, _ = pickle_safe_ram_hp(data)
        usr1, _ = unpickle_safe_ram()

        # sq, _ = send_queue(data)
        # gq, _ = get_queue()

        with open(f"logs/writers1.log", "a+") as log:
            log.write(
                f"{psd},{psr},{psrh}\n")

        with open(f"logs/readers1.log", "a+") as red:
            red.write(
                f"{usd},{usr},{usr1}\n")

"""
# # Numpy

# start = time.process_time()
# new_data = {}

# for k in data:
#     new_key = k
#     if isinstance(data[k],dict):
#         for p in data[k]:
#             if isinstance(data[k][p], dict):
#                 for y in data[k][p]:
#                     new_key = f'{k}.{p}.{y}'
#                     new_data[new_key]=data[k][p][y]

# for n in range(REPETITIONS):
#     for k in data:
#         if isinstance(data[k],dict):
#             with open(f"/dev/shm/test.{k}.bin", "wb") as data_file:
#                     np.savez(data_file, **data[k])
#         else:
#             with open(f"/dev/shm/test.{k}.bin", "wb") as data_file:
#                     np.savez(data_file, data[k])

# print(f"numpy+RAM:        {(time.process_time()-start)/REPETITIONS}")
"""
