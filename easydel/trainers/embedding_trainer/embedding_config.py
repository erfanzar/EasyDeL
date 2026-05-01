# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the EmbeddingTrainer."""

from dataclasses import dataclass, field
from typing import Literal

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "embedding")
@dataclass
class EmbeddingConfig(TrainingArguments):
    """Configuration for contrastive embedding model training.

    Supports multiple contrastive learning objectives for training
    dense text embedding models:

    - **InfoNCE** (default): in-batch negatives with temperature-scaled
      cross-entropy. The standard objective used by E5, GTE, BGE.
    - **Triplet**: margin-based triplet loss on
      ``(anchor, positive, negative)`` triples.
    - **MNRL** (Multiple Negatives Ranking Loss): variant of InfoNCE
      tailored to retrieval, equivalent to InfoNCE with cosine
      similarity.

    The trainer expects datasets with at minimum ``query`` /
    ``positive`` columns; for triplet or hard-negative training a
    ``negative`` column may also be supplied. Optionally, when
    ``matryoshka_dims`` is set, the contrastive loss is evaluated at
    each requested embedding dimension and averaged
    (Matryoshka Representation Learning).

    Construct with dict-literal kwargs:

    >>> cfg = EmbeddingConfig(loss_type="infonce", temperature=0.05,
    ...                       max_length=512, total_batch_size=128)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"Embedding"``).
        loss_type: Contrastive objective. One of ``"infonce"``,
            ``"triplet"``, ``"mnrl"``. Default ``"infonce"``.
        temperature: Logit-scale temperature applied before the
            softmax in InfoNCE/MNRL. Lower values produce sharper
            distributions. Typical range ``0.01-0.1``.
        margin: Triplet loss margin. Ignored for InfoNCE/MNRL.
        query_field: Dataset column carrying anchor/query strings.
        positive_field: Dataset column carrying positive strings.
        negative_field: Optional dataset column carrying hard
            negatives. ``None`` falls back to in-batch negatives only.
        max_length: Maximum tokenized sequence length.
        matryoshka_dims: Optional list of embedding sub-dimensions for
            MRL. The loss is computed at each dimension and averaged.
        normalize_embeddings: When ``True``, L2-normalises embeddings
            before computing similarity (must match the model's
            inference-time setting).
        pooling_strategy: Override for the model's pooling strategy
            (``"last"``, ``"mean"``, ``"first"``, ``"weighted_mean"``,
            ``"max"``). ``None`` keeps the model default.
        dataset_num_proc: Worker count for ``Dataset.map`` during
            preprocessing.
    """

    trainer_prefix: str | None = field(
        default="Embedding",
        metadata={"help": "Prefix for trainer logging."},
    )

    loss_type: Literal["infonce", "triplet", "mnrl"] = field(
        default="infonce",
        metadata={
            "help": (
                "Contrastive loss function: 'infonce' (in-batch negatives with "
                "temperature-scaled cross-entropy), 'triplet' (margin-based), or "
                "'mnrl' (multiple negatives ranking loss, equivalent to InfoNCE "
                "with cosine similarity)."
            )
        },
    )

    temperature: float = field(
        default=0.05,
        metadata={
            "help": (
                "Temperature for scaling similarity logits in InfoNCE/MNRL. "
                "Lower = sharper distribution. Typical range: 0.01-0.1."
            )
        },
    )

    margin: float = field(
        default=0.2,
        metadata={"help": "Margin for triplet loss. Ignored for InfoNCE/MNRL."},
    )

    query_field: str = field(
        default="query",
        metadata={"help": "Dataset column containing query/anchor texts."},
    )

    positive_field: str = field(
        default="positive",
        metadata={"help": "Dataset column containing positive/similar texts."},
    )

    negative_field: str | None = field(
        default=None,
        metadata={"help": ("Dataset column containing hard negative texts. If None, only in-batch negatives are used.")},
    )

    max_length: int | None = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."},
    )

    matryoshka_dims: list[int] | None = field(
        default=None,
        metadata={
            "help": (
                "List of embedding dimensions for Matryoshka Representation "
                "Learning. Loss is computed at each dim and averaged. "
                "Example: [64, 128, 256, 768]."
            )
        },
    )

    normalize_embeddings: bool = field(
        default=True,
        metadata={"help": "L2-normalize embeddings before computing similarity."},
    )

    pooling_strategy: str | None = field(
        default=None,
        metadata={
            "help": (
                "Pooling strategy override ('last', 'mean', 'first', "
                "'weighted_mean', 'max'). None uses the model's default."
            )
        },
    )

    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )

    __hash__ = hash_fn
