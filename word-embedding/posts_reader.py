class PostsReader:
    def __init__(self, data):
        self.data = data
        self.iterator = filter(lambda x: x == x, data)
        self.epoch = 0

    @property
    def iterator(self):
        return self.__iter

    @iterator.setter
    def iterator(self, value):
        self.__iter = value

    @property
    def epoch(self):
        return self.__epoch

    @epoch.setter
    def epoch(self, value):
        if value < 0:
            self.__epoch = 0
        else:
            self.__epoch = value

    def read_posts(self, idx):
        if idx == 0:
            self.epoch += 1
            print(f"Epoch: {self.epoch}")
            self.iterator = filter(lambda x: x == x, self.data)
        try:
            return next(self.iterator)
        except StopIteration:
            return None
