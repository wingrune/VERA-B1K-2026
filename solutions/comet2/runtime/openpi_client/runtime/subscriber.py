import abc


class Subscriber(abc.ABC):


    @abc.abstractmethod
    def on_episode_start(self) -> None:
        pass

    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        pass

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        pass
