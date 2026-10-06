class Registry:
    __registry = {}

    def __init__(self, name):
        self.name = name

    def register(self, key: str, value: object) -> None:
        self.__registry[key] = value

    def get(self, key: str) -> object:
        assert (
            key in self.__registry
        ), f"Key {key} is no registered in registry {self.name}"
        return self.__registry[key]


factories = Registry("factories")
