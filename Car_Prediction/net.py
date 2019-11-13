import abc


class Net(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def setup_model(self):
        return
