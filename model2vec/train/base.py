from __future__ import annotations

import logging
from typing import Any, TypeVar

import numpy as np
import torch
from tokenizers import Encoding, Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from model2vec import StaticModel

logger = logging.getLogger(__name__)


class FinetunableStaticModel(nn.Module):
    def __init__(self, *, vectors: torch.Tensor, tokenizer: Tokenizer, out_dim: int = 2, pad_id: int = 0) -> None:
        """
        Initialize a trainable StaticModel from a StaticModel.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param out_dim: The output dimension of the head.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models
        """
        super().__init__()
        self.pad_id = pad_id
        self.out_dim = out_dim
        self.embed_dim = vectors.shape[1]

        self.vectors = vectors
        if self.vectors.dtype != torch.float32:
            dtype = str(self.vectors.dtype)
            logger.warning(
                f"Your vectors are {dtype} precision, converting to to torch.float32 to avoid compatibility issues."
            )
            self.vectors = vectors.float()

        self.embeddings = nn.Embedding.from_pretrained(vectors.clone(), freeze=False, padding_idx=pad_id)
        self.head = self.construct_head()
        self.w = self.construct_weights()
        self.tokenizer = tokenizer

    def construct_weights(self) -> nn.Parameter:
        """Construct the weights for the model."""
        weights = torch.zeros(len(self.vectors))
        weights[self.pad_id] = -10_000
        return nn.Parameter(weights)

    def construct_head(self) -> nn.Sequential:
        """Method should be overridden for various other classes."""
        return nn.Sequential(nn.Linear(self.embed_dim, self.out_dim))

    @classmethod
    def from_pretrained(
        cls: type[ModelType], *, out_dim: int = 2, model_name: str = "minishlab/potion-base-32m", **kwargs: Any
    ) -> ModelType:
        """Load the model from a pretrained model2vec model."""
        model = StaticModel.from_pretrained(model_name)
        return cls.from_static_model(model=model, out_dim=out_dim, **kwargs)

    @classmethod
    def from_static_model(cls: type[ModelType], *, model: StaticModel, out_dim: int = 2, **kwargs: Any) -> ModelType:
        """Load the model from a static model."""
        model.embedding = np.nan_to_num(model.embedding)
        embeddings_converted = torch.from_numpy(model.embedding)
        return cls(
            vectors=embeddings_converted,
            pad_id=model.tokenizer.token_to_id("[PAD]"),
            out_dim=out_dim,
            tokenizer=model.tokenizer,
            **kwargs,
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        A forward pass and mean pooling.

        This function is analogous to `StaticModel.encode`, but reimplemented to allow gradients
        to pass through.

        :param input_ids: A 2D tensor of input ids. All input ids are have to be within bounds.
        :return: The mean over the input ids, weighted by token weights.
        """
        w = self.w[input_ids]
        w = torch.sigmoid(w)
        zeros = (input_ids != self.pad_id).float()
        w = w * zeros
        # Add a small epsilon to avoid division by zero
        length = zeros.sum(1) + 1e-16
        embedded = self.embeddings(input_ids)
        # Weigh each token
        embedded = torch.bmm(w[:, None, :], embedded).squeeze(1)
        # Mean pooling by dividing by the length
        embedded = embedded / length[:, None]

        return nn.functional.normalize(embedded)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        encoded = self._encode(input_ids)
        return self.head(encoded), encoded

    def tokenize(self, texts: list[str], max_length: int | None = 512) -> torch.Tensor:
        """
        Tokenize a bunch of strings into a single padded 2D tensor.

        Note that this is not used during training.

        :param texts: The texts to tokenize.
        :param max_length: If this is None, the sequence lengths are truncated to 512.
        :return: A 2D padded tensor
        """
        encoded: list[Encoding] = self.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        encoded_ids: list[torch.Tensor] = [torch.Tensor(encoding.ids[:max_length]).long() for encoding in encoded]
        return pad_sequence(encoded_ids, batch_first=True, padding_value=self.pad_id)

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return self.embeddings.weight.device

    def to_static_model(self) -> StaticModel:
        """Convert the model to a static model."""
        emb = self.embeddings.weight.detach().cpu().numpy()
        w = torch.sigmoid(self.w).detach().cpu().numpy()

        return StaticModel(emb * w[:, None], self.tokenizer, normalize=True)


@dataclass
class EnsembleComponent:
    vectors: torch.Tensor
    tokenizer: Tokenizer
    pad_id: int = 0


class FinetunableStaticEnsembleModel(nn.Module):
    def __init__(self, *, components: list[EnsembleComponent], out_dim: int = 2) -> None:
        """
        Initialize a trainable StaticModel from a StaticModel.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param out_dim: The output dimension of the head.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models
        """
        super().__init__()
        self.device = self.get_device()
        self.out_dim = out_dim
        self.components = components
        for c_id, component in enumerate(components):
            self.components[c_id].vectors = component.vectors.to(self.device)
        self.embeddings = [
            nn.Embedding(
                num_embeddings=component.vectors.shape[0],
                embedding_dim=component.vectors.shape[1],
                padding_idx=component.pad_id,
                _weight=component.vectors,
                _freeze=False,
            ) for component in self.components
        ]
        self.embed_dim = [component.vectors.shape[1] for component in components]

        for component in self.components: 
            if component.vectors.dtype != torch.float32:
                dtype = str(component.vectors.dtype)
                logger.warning(
                    f"Your vectors are {dtype} precision, converting to to torch.float32 to avoid compatibility issues."
                )
                component.vectors = component.vectors.float()

        self.head = self.construct_head()
        self.w = self.construct_weights()

    def get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def construct_weights(self) -> list[nn.Parameter]:
        """Construct the weights for the model."""
        weights = []
        for component in self.components:
            component_weights = torch.zeros(
                len(component.vectors),
                device=self.device,
            )
            component_weights[component.pad_id] = -10_000
            weights.append(nn.Parameter(component_weights))
        return weights

    def construct_head(self) -> nn.Sequential:
        """Method should be overridden for various other classes."""
        return nn.Sequential(nn.Linear(sum(self.embed_dim), self.out_dim))

    @classmethod
    def from_pretrained(
        cls: type[ModelType], *, out_dim: int = 2, model_names: list[str] = ["minishlab/potion-base-2M", "minishlab/potion-base-4M", "minishlab/potion-base-8M"], **kwargs: Any
    ) -> ModelType:
        """Load the models from pretrained model2vec models."""
        models = [StaticModel.from_pretrained(model_name) for model_name in model_names]
        return cls.from_static_model(models=models, out_dim=out_dim, **kwargs)

    @classmethod
    def from_static_model(cls: type[ModelType], *, models: list[StaticModel], out_dim: int = 2, **kwargs: Any) -> ModelType:
        """Load the ensemble model from static models."""
        components = []
        for model in models:
            model.embedding = np.nan_to_num(model.embedding)
            embeddings_converted = torch.from_numpy(model.embedding)
            components.append(
                EnsembleComponent(
                    vectors=embeddings_converted,
                    tokenizer=model.tokenizer,
                    pad_id=model.tokenizer.token_to_id("[PAD]")
                )
            )
        return cls(
            components=components,
            out_dim=out_dim,
            **kwargs,
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        A forward pass and mean pooling.

        This function is analogous to `StaticModel.encode`, but reimplemented to allow gradients
        to pass through.

        :param input_ids: A 3D tensor of input ids. All input ids are have to be within bounds.
        :return: The mean over the input ids, weighted by token weights.
        """
        embedded = []
        in_cpu = input_ids.is_cpu
        if in_cpu:
            input_ids = input_ids.to(self.device)
        for c_id, component in enumerate(self.components):
            component_input_ids = input_ids[:, :, c_id]
            w = self.w[c_id][component_input_ids]
            w = torch.sigmoid(w)
            zeros = (component_input_ids != component.pad_id).float()
            w = w * zeros
            # Add a small epsilon to avoid division by zero
            length = zeros.sum(1) + 1e-16
            component_embedded = self.embeddings[c_id](component_input_ids)
            # Weigh each token
            component_embedded = torch.bmm(w[:, None, :], component_embedded).squeeze(1)
            # Mean pooling by dividing by the length
            component_embedded = component_embedded / length[:, None]
            embedded.append(component_embedded)

        normalized = nn.functional.normalize(torch.cat(embedded, dim=1))
        if in_cpu:
            normalized = normalized.to("cpu")
        return normalized

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        encoded = self._encode(input_ids)
        return self.head(encoded), encoded

    def tokenize(self, texts: list[str], max_length: int | None = 512) -> torch.Tensor:
        """
        Tokenize a bunch of strings into a single padded 2D tensor.

        Note that this is not used during training.

        :param texts: The texts to tokenize.
        :param max_length: If this is None, the sequence lengths are truncated to 512.
        :return: A 3D padded tensor
        """
        encoded = []
        for component in self.components:
            component_encoded: list[Encoding] = component.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
            component_encoded_ids: list[torch.Tensor] = [torch.Tensor(encoding.ids[:max_length]).long() for encoding in component_encoded]
            component_padded = pad_sequence(component_encoded_ids, batch_first=True, padding_value=component.pad_id)
            encoded.append(component_padded.T)
        return pad_sequence(encoded, batch_first=True, padding_value=0).swapaxes(0, 2)

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return self.embeddings[0].weight.device

    def to_static_models(self) -> list[StaticModel]:
        """Convert the model to a static model."""
        models = []
        for c_id, component in enumerate(self.components):
            emb = self.embeddings[c_id].weight.detach().cpu().numpy()
            w = torch.sigmoid(self.w[c_id]).detach().cpu().numpy()
            models.append(
                StaticModel(emb * w[:, None], component.tokenizer, normalize=True)
            )
        return models


class TextDataset(Dataset):
    def __init__(self, tokenized_texts: list[list[int]], targets: torch.Tensor) -> None:
        """
        A dataset of texts.

        :param tokenized_texts: The tokenized texts. Each text is a list of token ids.
        :param targets: The targets.
        :raises ValueError: If the number of labels does not match the number of texts.
        """
        if len(targets) != len(tokenized_texts):
            raise ValueError("Number of labels does not match number of texts.")
        self.tokenized_texts = tokenized_texts
        self.targets = targets

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tokenized_texts)

    def __getitem__(self, index: int) -> tuple[list[int], torch.Tensor]:
        """Gets an item."""
        return self.tokenized_texts[index], self.targets[index]

    @staticmethod
    def collate_fn(batch: list[tuple[list[list[int]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors = [torch.LongTensor(x) for x in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)

        return padded, torch.stack(targets)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)


class EnsembleTextDataset(Dataset):
    def __init__(self, tokenized_texts: list[list[list[int]]], targets: torch.Tensor) -> None:
        """
        A dataset of texts.

        :param tokenized_texts: The tokenized texts. Each text is a list of token ids.
        :param targets: The targets.
        :raises ValueError: If the number of labels does not match the number of texts.
        """
        if any([len(targets) != len(c_tokenized_texts) for c_tokenized_texts in tokenized_texts]):
            raise ValueError("Number of labels does not match number of texts.")
        self.tokenized_texts = tokenized_texts
        self.targets = targets

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[list[list[int]], torch.Tensor]:
        """Gets an item."""
        return [c_tokenized_texts[index] for c_tokenized_texts in self.tokenized_texts], self.targets[index]

    @staticmethod
    def collate_fn(batch: list[tuple[list[list[list[int]]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors = []
        for component_texts in texts:
            component_tensors = [torch.LongTensor(x) for x in component_texts]
            component_padded = pad_sequence(component_tensors, batch_first=True, padding_value=0)
            tensors.append(component_padded.T)

        padded = pad_sequence(tensors, batch_first=True, padding_value=0)

        return padded, torch.stack(targets)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)


ModelType = TypeVar("ModelType", bound=FinetunableStaticModel)
