import multiprocessing as mp

class Port():
    def __init__(self):
        self.queue = mp.Queue()

    def put(self, item):
        self.queue.put(item)

    def get(self):
        return self.queue.get()

    def get_buffered(self, num_items):
        list = []
        while len(list) < num_items:
            list += self.queue.get()
        return list