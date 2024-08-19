"""🌌 stream abstractions."""
# ruff: noqa: ERA001

from typing import Any

import numpy as np
from beartype import beartype
from loguru import logger

from streamgen import is_extra_installed
from streamgen.samplers import Sampler

if is_extra_installed("cl"):
    import torch

    #! avalanche-lib has a broken version constraint on torchcv
    # * because I do not want to wait for the fix and since these mehtods
    # * only provide application-specific starting points, I decided to exclude them temporarily
    # from avalanche.benchmarks.utils import as_classification_dataset
    # from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
    from continuum.datasets import InMemoryDataset
    from continuum.scenarios.continual_scenario import ContinualScenario
    from torch.utils.data import TensorDataset

    @beartype
    def collect_stream(
        sampler: Sampler,
        n_experiences: int,
        n_samples_per_experience: int = 100,
    ) -> list[Any]:
        """🌌 collects a stream of `n_experiences` with `n_samples_per_experience` by calling `sampler.collect`.

        Args:
            sampler (Sampler): data generator
            n_experiences (int, optional): Number of experiences. Each experience represents a different parameterization of the sampler.
            n_samples_per_experience (int, optional): Number of samples to collect in each experience. Defaults to 100

        Returns:
            list[Any]: a list of experiences. The format and type of each experience depends on the `collect` implementation of the sampler.
        """
        logger.info(f"🌌 collecting {n_experiences} experiences with {n_samples_per_experience} samples each.")

        return [sampler.collect(n_samples_per_experience) for _ in range(n_experiences)]

    @beartype
    def to_tensor_dataset(samples: np.ndarray, targets: np.ndarray) -> TensorDataset:
        """🗃️ constructs a TensorDataset with a `targets` field from two numpy arrays."""
        samples = torch.tensor(samples)
        targets = torch.tensor(targets)

        dataset = TensorDataset(samples, targets)
        dataset.targets = targets

        return dataset

    # @beartype
    # def construct_avalanche_classification_datasets(experiences: list[tuple[np.ndarray, np.ndarray]]) -> list[ClassificationDataset]:
    #     """❄️ constructs a list of avalanche `ClassificationDataset`s.

    #     This can be used with `avalanche.benchmarks.scenarios.dataset_scenario.benchmark_from_datasets` to create an avalanche benchmark.

    #     Args:
    #         experiences (list[tuple[np.ndarray, np.ndarray]]): list of experiences.

    #     Returns:
    #         list[ClassificationDataset]: list of avalanche `ClassificationDataset`s.
    #     """
    #     stream = []

    #     for experience in experiences:
    #         samples, targets = experience

    #         dataset = to_tensor_dataset(samples, targets)

    #         dataset = as_classification_dataset(dataset)

    #         stream.append(dataset)

    #     return stream

    @beartype
    def construct_continuum_scenario(experiences: list[tuple[np.ndarray, np.ndarray]]) -> ContinualScenario:
        """🕑 constructs a continuum `ContinualScenario`.

        Args:
            experiences (list[tuple[np.ndarray, np.ndarray]]): list of experiences.

        Returns:
            ContinualScenario: generic continuum scenario.
        """
        x, y, t = [], [], []

        for idx, experience in enumerate(experiences):
            samples, targets = experience
            task_ids = np.ones_like(targets) * idx
            x.append(samples)
            y.append(targets)
            t.append(task_ids)

        x = np.concatenate(x)
        y = np.concatenate(y)
        t = np.concatenate(t)

        dataset = InMemoryDataset(x, y, t)

        return ContinualScenario(dataset)
