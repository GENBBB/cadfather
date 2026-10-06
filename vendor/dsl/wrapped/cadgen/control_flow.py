import logging

from copy import deepcopy

import numpy as np

from .base import BaseFactory
from .registry import factories as factories_registry

logger = logging.getLogger(__name__)


def expand_factory_entries(factory_entries: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for entry in factory_entries:
        n_times = int(entry.get("n_times", 1))
        if n_times < 1:
            raise ValueError("n_times must be >= 1")

        for _ in range(n_times):
            repeated = deepcopy(entry)
            repeated.pop("n_times", None)
            expanded.append(repeated)
    return expanded


class EitherFactory(BaseFactory):
    def __init__(
        self,
        factories: list[dict[int, dict[str, BaseFactory]]],
        probabilities: list[float],
    ):
        self.factories = factories
        self.probabilities = probabilities
        self.build_factories()

    def build_factories(self):
        factories_built = []

        for factories in self.factories:
            _factories_built = []
            for value in factories.values():
                factory_name = list(value.keys())[0]
                factory_kwargs = value[factory_name]
                factory = factory_kwargs.pop("factory")  # type: ignore
                _factories_built.append(
                    dict(
                        factory=factories_registry.get(factory_name)(**factory),  # type: ignore
                        **factory_kwargs,  # type: ignore
                    )
                )
            factories_built.append(expand_factory_entries(_factories_built))

        self.factories = factories_built

    def generate(self):
        chosen_factory_index = np.random.choice(
            np.arange(len(self.factories)), p=self.probabilities
        )
        chosen_factories = self.factories[chosen_factory_index]
        return chosen_factories


class BlockFactory(BaseFactory):
    def __init__(self, factories: list[dict[int, dict[str, BaseFactory]]]):
        self.factories = factories
        self.build_factories()

    def build_factories(self):
        factories_built = []

        for factory in self.factories:
            factory_name = list(factory.keys())[0]
            factory_kwargs = factory[factory_name]
            factory = factory_kwargs.pop("factory")  # type: ignore
            factories_built.append(
                dict(
                    factory=factories_registry.get(factory_name)(**factory),  # type: ignore
                    **factory_kwargs,  # type: ignore
                )
            )
        self.factories = expand_factory_entries(factories_built)

    def generate(self):
        pass


class StopFactory(BaseFactory):
    def __init__(self):
        pass

    def generate(self):
        pass


factories_registry.register("either", EitherFactory)
factories_registry.register("block", BlockFactory)
factories_registry.register("stop", StopFactory)
