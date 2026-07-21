import random
import time
from types import TracebackType
from typing import Self


class Timer:
    """Measure the execution time of a code block."""

    def __enter__(self) -> Self:
        """Start timing."""

        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop timing."""

        self.elapsed = time.perf_counter() - self._start


def smoke_test():
    with Timer() as timer:
        print("Doing something important that takes time...")
        time.sleep(random.uniform(0.5, 1.5))
        print("Finished important task.")

    print(f"Elapsed time: {timer.elapsed:.3f} seconds")


if __name__ == "__main__":
    smoke_test()
