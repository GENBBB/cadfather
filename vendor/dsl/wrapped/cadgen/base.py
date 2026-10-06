from abc import ABC, abstractmethod


class BaseOperation(ABC):
    """Abstract base class for CAD operations."""

    @abstractmethod
    def to_string(self, *args, **kwargs) -> str:  # pragma: no cover - interface only
        """Serialize operation to CadQuery expression."""

    @abstractmethod
    def transform(  # pragma: no cover - interface only
        self, shift: list[float], scale: float, *args, **kwargs
    ) -> None:
        """Apply global shift/scale to operation geometry."""

    @abstractmethod
    def round(self) -> None:  # pragma: no cover - interface only
        """Round numeric parameters to integers."""

    @abstractmethod
    def fix(self) -> None:  # pragma: no cover - interface only
        """Fix operation geometry."""


class BaseFactory(ABC):
    """Abstract base class for random operation generators."""

    @abstractmethod
    def generate(self) -> BaseOperation:  # pragma: no cover - interface only
        """Produce a new operation instance."""
