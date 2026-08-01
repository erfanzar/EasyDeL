# Docstring Style Reference

Concrete templates and worked examples for the swarm in `SKILL.md`. Detect the
project's existing convention and match it; default to **Google** if none
dominates. Never mix styles within a file.

## Detecting the convention

- **Google** — `Args:`, `Returns:`, `Raises:` headers, fields as
  `name (type): desc`.
- **NumPy** — `Parameters`, `Returns` headers underlined with `----`, fields as
  `name : type` then an indented description.
- **reST / Sphinx** — `:param name:`, `:type name:`, `:returns:`, `:raises:`.

```bash
rg -n ':param |:returns:' <target>   # reST
rg -n '^\s*Parameters\s*$' <target>  # NumPy (underlined)
rg -n '^\s*Args:\s*$' <target>       # Google
```

## Module / file docstring

```python
"""Short statement of what this module is for.

Longer paragraph: the main classes/functions it provides, how it fits the
package, and any important usage or invariants a reader needs up front.
"""
```

Place it as the **very first statement** of the file, before imports.

## Function — Google style (default)

```python
def resize_image(image, size, *, keep_aspect=True, fill=0):
    """Resize an image to a target size.

    Resamples `image` to `size`. When `keep_aspect` is True the image is scaled
    to fit within `size` and the remainder is padded with `fill`.

    Args:
        image (np.ndarray): Source image, shape (H, W, C), dtype uint8.
        size (tuple[int, int]): Target (height, width) in pixels.
        keep_aspect (bool): If True, preserve aspect ratio and pad; if False,
            stretch to exactly `size`. Defaults to True.
        fill (int): Pad value used when `keep_aspect` is True. Defaults to 0.

    Returns:
        np.ndarray: Resized image, shape (size[0], size[1], C), dtype uint8.

    Raises:
        ValueError: If `image` is not 3-dimensional.
    """
```

`IN` = `Args:`, `OUT` = `Returns:` (use `Yields:` for generators). Document
`*args`/`**kwargs` as their own entries.

## Function — NumPy style

```python
def resize_image(image, size, *, keep_aspect=True, fill=0):
    """Resize an image to a target size.

    Parameters
    ----------
    image : np.ndarray
        Source image, shape (H, W, C), dtype uint8.
    size : tuple of int
        Target (height, width) in pixels.
    keep_aspect : bool, default True
        Preserve aspect ratio and pad if True; stretch if False.
    fill : int, default 0
        Pad value used when ``keep_aspect`` is True.

    Returns
    -------
    np.ndarray
        Resized image, shape (size[0], size[1], C), dtype uint8.

    Raises
    ------
    ValueError
        If ``image`` is not 3-dimensional.
    """
```

## Class

Document the class purpose and its notable attributes **in the class docstring**
— never as string literals after attribute assignments.

```python
class Cache:
    """Fixed-capacity LRU cache.

    Stores up to `capacity` items, evicting the least-recently-used entry on
    overflow. Not thread-safe.

    Attributes:
        capacity (int): Maximum number of items retained.
        hits (int): Number of successful lookups since construction.
        misses (int): Number of failed lookups since construction.
    """

    def __init__(self, capacity):
        """Initialize the cache.

        Args:
            capacity (int): Maximum number of items to retain; must be > 0.

        Raises:
            ValueError: If `capacity` is not positive.
        """
```

## Expanding `**kwargs: Unpacked[TypedDict]` (PEP 692)

Given:

```python
class TrainKwargs(TypedDict, total=False):
    lr: float           # learning rate
    warmup: int         # warmup steps
    clip: float | None  # gradient clip norm, or None to disable

def train(model, data, **kwargs: Unpacked[TrainKwargs]):
    ...
```

Open `TrainKwargs`, read its fields, and document **each one** as a keyword
argument — do not leave `**kwargs` opaque:

```python
def train(model, data, **kwargs: Unpacked[TrainKwargs]):
    """Train `model` on `data`.

    Args:
        model (Module): The model to update in place.
        data (Iterable[Batch]): Training batches.

    Keyword Args:
        lr (float): Learning rate. Passed via **kwargs (TrainKwargs).
        warmup (int): Number of warmup steps before the schedule. Via **kwargs.
        clip (float | None): Gradient clip norm; None disables clipping. Via
            **kwargs.

    Returns:
        TrainState: Final training state after the last batch.
    """
```

If any unpacked field changes the return (e.g. a `return_history` flag), note
that in `Returns:` as well.

## Updating an outdated docstring

Drift to fix: renamed/removed params, changed return type, stale prose. Example
— the signature dropped `verbose` and now returns a count, not a bool:

```python
# stale:                                  # corrected:
"""Run the job.                           """Run the job.

Args:                                     Args:
    path (str): Input path.                   path (str): Input path.
    verbose (bool): Print progress.       Returns:
                                              int: Number of records processed.
Returns:                                  """
    bool: True on success.
"""
```

Only rewrite the parts that are wrong; keep accurate prose intact to minimize
diff noise.

## The forbidden pattern — never introduce this

```python
LEARNING_RATE = 3e-4
"""The default learning rate."""          # ← DO NOT ADD. Variable/attribute
                                          #   string-literal docstrings are out
                                          #   of scope at every level.
```

Instead, if the constant is worth documenting, mention it in the **module
docstring** (or the class `Attributes:` section for class attributes). Leave
the assignment itself untouched.
