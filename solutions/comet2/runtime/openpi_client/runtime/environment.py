import abc


class Environment(abc.ABC):


    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict:
        pass

    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        pass
