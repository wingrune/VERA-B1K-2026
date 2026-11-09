import abc


class Agent(abc.ABC):


    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass
