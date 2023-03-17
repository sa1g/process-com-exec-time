import matplotlib.pyplot as plt
import pandas

if __name__ == '__main__':
    WRITERS = 'logs/writers.log'
    READERS = 'logs/readers.log'
    iterations = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]

    # WRITERS

    writers = pandas.read_csv(WRITERS)
    default_x_ticks = range(len(iterations))
    # plt.plot(default_x_ticks, writers['pickle_unsafe_disk'].to_numpy()  , label="pickle_unsafe_disk")
    # plt.plot(default_x_ticks, writers['pickle_unsafe_ram'].to_numpy()   , label="pickle_unsafe_ram")
    # plt.plot(default_x_ticks, writers['pickle_unsafe_ram_hp'].to_numpy(), label="pickle_unsafe_ram_hp")
    # plt.plot(default_x_ticks, writers['dill_unsafe_disk'].to_numpy()    , label="dill_unsafe_disk")
    # plt.plot(default_x_ticks, writers['dill_unsafe_ram'].to_numpy()     , label="dill_unsafe_ram")
    plt.plot(default_x_ticks, writers['pickle_safe_disk'].to_numpy()    , label="pickle_safe_disk")
    plt.plot(default_x_ticks, writers['pickle_safe_ram'].to_numpy()     , label="pickle_safe_ram")
    plt.plot(default_x_ticks, writers['pickle_safe_ram_hp'].to_numpy()  , label="pickle_safe_ram_hp")
    # plt.plot(default_x_ticks, writers['send_queue'].to_numpy()          , label="send_queue")

    plt.xticks(default_x_ticks, iterations)

    plt.xlabel("Batch size")
    plt.ylabel("Batch sending execution time [s]")
    plt.title("writers")

    plt.legend()
    plt.gcf().set_size_inches(10, 7)

    plt.savefig("plots/writers.png", dpi=400)
    plt.close()


    # READERS
    readers = pandas.read_csv(READERS)
    default_x_ticks = range(len(iterations))
    # plt.plot(default_x_ticks, readers['unpickle_unsafe_disk'].to_numpy()  , label="unpickle_unsafe_disk")
    # plt.plot(default_x_ticks, readers['unpickle_unsafe_ram'].to_numpy()   , label="unpickle_unsafe_ram")
    # plt.plot(default_x_ticks, readers['unpickle_unsafe_ram_hp'].to_numpy(), label="unpickle_unsafe_ram_hp")
    # plt.plot(default_x_ticks, readers['undill_unsafe_disk'].to_numpy()    , label="undill_unsafe_disk")
    # plt.plot(default_x_ticks, readers['undill_unsafe_ram'].to_numpy()     , label="undill_unsafe_ram")
    plt.plot(default_x_ticks, readers['unpickle_safe_disk'].to_numpy()    , label="unpickle_safe_disk")
    plt.plot(default_x_ticks, readers['unpickle_safe_ram'].to_numpy()     , label="unpickle_safe_ram")
    plt.plot(default_x_ticks, readers['unpickle_safe_ram_hp'].to_numpy()  , label="unpickle_safe_ram_hp")
    # plt.plot(default_x_ticks, readers['get_queue'].to_numpy()          , label="get_queue")

    plt.xticks(default_x_ticks, iterations)

    plt.xlabel("Batch size")
    plt.ylabel("Batch receiving execution time [s]")
    plt.title("readers")

    plt.legend()
    plt.gcf().set_size_inches(10, 7)

    plt.savefig("plots/readers.png", dpi=400)
    plt.close()
