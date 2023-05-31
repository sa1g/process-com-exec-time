import matplotlib.pyplot as plt
import pandas

if __name__ == "__main__":
    iterations = [1, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]#, 650, 750, 1000, 1500, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    
    writers = [
        "pickle_safe_ram",
        "pickle_safe_ram_hp",
        "dill_safe_ram",
        "dill_safe_ram_hp",
    ]

    readers = [
        "unpickle_safe_ram",
        "unpickle_safe_ram_hp",
        "undill_safe_ram",
        "undill_safe_ram_hp",
    ]

    # Load files:
    writers_files = {name: pandas.read_csv(
        f"logs/{name}.csv") for name in writers}
    readers_files = {name: pandas.read_csv(
        f"logs/{name}.csv") for name in readers}

    # WRITERS PLOT
    default_x_ticks = range(len(iterations))

    # SCOMPATTATO
    for key, key1 in zip(readers_files.keys(), writers_files.keys()):
        if key1 not in ['send_queue', 'dill_unsafe_disk', 'pickle_safe_disk', 'pickle_unsafe_disk', 'dill_unsafe_ram']:
            plt.scatter(iterations, readers_files[key].mean(1), label=key1)
        # plt.plot(iterations, readers_files[key].mean(1)*100  , label=key)
        plt.xticks(iterations)

    plt.style.use('default')

    plt.xticks(rotation=-45)
    plt.xlabel("Batch size")
    plt.ylabel("Communication time - send data [s]")
    # plt.title("Receive (12) and Send (1 - as it's concurrent on 12 processes) data - no some")
    plt.title("Receive data")

    plt.legend()
    plt.gcf().set_size_inches(10, 7)
    plt.savefig("plot/tmp121.png", dpi=400)
    plt.close()
