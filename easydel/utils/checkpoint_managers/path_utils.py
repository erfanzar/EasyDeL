# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import io
import json
import os
import pickle
import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from google.cloud import storage


class UniversalPath(ABC):
    """Abstract base class for universal path operations"""

    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def read_text(self, encoding: str = "utf-8") -> str:
        pass

    @abstractmethod
    def write_text(self, data: str, encoding: str = "utf-8") -> None:
        pass

    @abstractmethod
    def read_bytes(self) -> bytes:
        pass

    @abstractmethod
    def write_bytes(self, data: bytes) -> None:
        pass

    @abstractmethod
    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        pass

    @abstractmethod
    def is_dir(self) -> bool:
        pass

    @abstractmethod
    def is_file(self) -> bool:
        pass

    @abstractmethod
    def iterdir(self) -> Iterator["UniversalPath"]:
        pass

    @abstractmethod
    def glob(self, pattern: str) -> Iterator["UniversalPath"]:
        pass

    @abstractmethod
    def __truediv__(self, other) -> "UniversalPath":
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def as_posix(self) -> str:
        """Return the string representation with forward slashes"""
        pass

    @abstractmethod
    def stem(self) -> str:
        """Return the final path component without its suffix"""
        pass

    @abstractmethod
    def suffixes(self) -> list[str]:
        """Return a list of the path's file suffixes"""
        pass

    @abstractmethod
    def with_name(self, name: str) -> "UniversalPath":
        """Return a new path with the name changed"""
        pass

    @abstractmethod
    def with_suffix(self, suffix: str) -> "UniversalPath":
        """Return a new path with the suffix changed"""
        pass

    @abstractmethod
    def with_stem(self, stem: str) -> "UniversalPath":
        """Return a new path with the stem changed"""
        pass

    @abstractmethod
    def parts(self) -> tuple[str, ...]:
        """Return a tuple of the path components"""
        pass

    @abstractmethod
    def relative_to(self, other: "UniversalPath") -> "UniversalPath":
        """Return a relative path from other to this path"""
        pass

    @abstractmethod
    def is_absolute(self) -> bool:
        """Return True if the path is absolute"""
        pass

    @abstractmethod
    def resolve(self) -> "UniversalPath":
        """Make the path absolute, resolving any symlinks"""
        pass

    @abstractmethod
    def rmdir(self) -> None:
        """Remove this directory (must be empty)"""
        pass

    @abstractmethod
    def unlink(self, missing_ok: bool = False) -> None:
        """Remove this file or symbolic link"""
        pass

    @abstractmethod
    def rename(self, target: "UniversalPath") -> "UniversalPath":
        """Rename this path to the given target"""
        pass

    @abstractmethod
    def stat(self) -> dict[str, Any]:
        """Return file statistics"""
        pass


class LocalPath(UniversalPath):
    """Local filesystem path wrapper around pathlib.Path"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.path.read_text(encoding=encoding)

    def write_text(self, data: str, encoding: str = "utf-8") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(data, encoding=encoding)

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def write_bytes(self, data: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        self.path.mkdir(parents=parents, exist_ok=exist_ok)

    def is_dir(self) -> bool:
        return self.path.is_dir()

    def is_file(self) -> bool:
        return self.path.is_file()

    def iterdir(self) -> Iterator["LocalPath"]:
        if self.path.is_dir():
            for item in self.path.iterdir():
                yield LocalPath(item)

    def glob(self, pattern: str) -> Iterator["LocalPath"]:
        for item in self.path.glob(pattern):
            yield LocalPath(item)

    def __truediv__(self, other) -> "LocalPath":
        return LocalPath(self.path / str(other))

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"LocalPath('{self.path}')"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def parent(self) -> "LocalPath":
        return LocalPath(self.path.parent)

    def as_posix(self) -> str:
        return self.path.as_posix()

    def stem(self) -> str:
        return self.path.stem

    def suffixes(self) -> list[str]:
        return self.path.suffixes

    def with_name(self, name: str) -> "LocalPath":
        return LocalPath(self.path.with_name(name))

    def with_suffix(self, suffix: str) -> "LocalPath":
        return LocalPath(self.path.with_suffix(suffix))

    def with_stem(self, stem: str) -> "LocalPath":
        return LocalPath(self.path.with_stem(stem))

    def parts(self) -> tuple[str, ...]:
        return self.path.parts

    def relative_to(self, other: "LocalPath") -> "LocalPath":
        if isinstance(other, LocalPath):
            return LocalPath(self.path.relative_to(other.path))
        else:
            return LocalPath(self.path.relative_to(Path(str(other))))

    def is_absolute(self) -> bool:
        return self.path.is_absolute()

    def resolve(self) -> "LocalPath":
        return LocalPath(self.path.resolve())

    def rmdir(self) -> None:
        self.path.rmdir()

    def unlink(self, missing_ok: bool = False) -> None:
        self.path.unlink(missing_ok=missing_ok)

    def rename(self, target: "LocalPath") -> "LocalPath":
        if isinstance(target, LocalPath):
            new_path = self.path.rename(target.path)
        else:
            new_path = self.path.rename(Path(str(target)))
        return LocalPath(new_path)

    def stat(self) -> dict[str, Any]:
        stat_result = self.path.stat()
        return {
            "size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "ctime": stat_result.st_ctime,
            "atime": stat_result.st_atime,
            "mode": stat_result.st_mode,
            "uid": stat_result.st_uid,
            "gid": stat_result.st_gid,
        }


class GCSPath(UniversalPath):
    """Google Cloud Storage path wrapper"""

    def __init__(self, path: str, client: storage.Client | None = None):
        if not path.startswith("gs://"):
            raise ValueError(f"GCS path must start with 'gs://': {path}")

        self.path = path
        self.client = client or storage.Client()

        path_parts = path[5:].split("/", 1)
        self.bucket_name = path_parts[0]
        self.blob_name = path_parts[1] if len(path_parts) > 1 else ""

        self._bucket = None
        self._blob = None

    @property
    def bucket(self):
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    @property
    def blob(self):
        if self._blob is None and self.blob_name:
            self._blob = self.bucket.blob(self.blob_name)
        return self._blob

    def exists(self) -> bool:
        if not self.blob_name:
            return self.bucket.exists()
        return self.blob.exists() if self.blob else False

    def read_text(self, encoding: str = "utf-8") -> str:
        if not self.blob:
            raise ValueError("Cannot read text from bucket root")
        return self.blob.download_as_text(encoding=encoding)

    def write_text(self, data: str, encoding: str = "utf-8") -> None:
        if not self.blob:
            raise ValueError("Cannot write text to bucket root")
        self.blob.upload_from_string(data, content_type="text/plain")

    def read_bytes(self) -> bytes:
        if not self.blob:
            raise ValueError("Cannot read bytes from bucket root")
        return self.blob.download_as_bytes()

    def write_bytes(self, data: bytes) -> None:
        if not self.blob:
            raise ValueError("Cannot write bytes to bucket root")
        self.blob.upload_from_string(data, content_type="application/octet-stream")

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        if self.blob_name and not self.blob_name.endswith("/"):
            placeholder_path = f"{self.blob_name}/"
        else:
            placeholder_path = self.blob_name or ""

        if placeholder_path:
            placeholder_blob = self.bucket.blob(placeholder_path + ".keep")
            if not placeholder_blob.exists():
                placeholder_blob.upload_from_string("", content_type="text/plain")

    def is_dir(self) -> bool:
        if not self.blob_name:
            return True

        prefix = self.blob_name if self.blob_name.endswith("/") else self.blob_name + "/"
        blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=1))
        return len(blobs) > 0

    def is_file(self) -> bool:
        return self.exists() and not self.is_dir()

    def iterdir(self) -> Iterator["GCSPath"]:
        if not self.blob_name:
            prefix = ""
            delimiter = "/"
        else:
            prefix = self.blob_name if self.blob_name.endswith("/") else self.blob_name + "/"
            delimiter = "/"

        for blob in self.bucket.list_blobs(prefix=prefix, delimiter=delimiter):
            if blob.name != prefix:
                yield GCSPath(f"gs://{self.bucket_name}/{blob.name}", self.client)

        for prfx in self.bucket.list_blobs(prefix=prefix, delimiter=delimiter).prefixes:
            yield GCSPath(f"gs://{self.bucket_name}/{prfx}", self.client)

    def glob(self, pattern: str) -> Iterator["GCSPath"]:
        import fnmatch

        prefix = self.blob_name if self.blob_name.endswith("/") else self.blob_name + "/"
        if not self.blob_name:
            prefix = ""

        for blob in self.bucket.list_blobs(prefix=prefix):
            relative_name = blob.name[len(prefix) :]
            if fnmatch.fnmatch(relative_name, pattern):
                yield GCSPath(f"gs://{self.bucket_name}/{blob.name}", self.client)

    def __truediv__(self, other) -> "GCSPath":
        other = str(other)
        if self.blob_name:
            new_path = f"{self.blob_name.rstrip('/')}/{other}"
        else:
            new_path = str(other)
        return GCSPath(f"gs://{self.bucket_name}/{new_path}", self.client)

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"GCSPath('{self.path}')"

    @property
    def name(self) -> str:
        if not self.blob_name:
            return self.bucket_name
        return os.path.basename(self.blob_name.rstrip("/"))

    @property
    def suffix(self) -> str:
        name = self.name
        return os.path.splitext(name)[1] if "." in name else ""

    @property
    def parent(self) -> "GCSPath":
        if not self.blob_name:
            raise ValueError("Bucket root has no parent")

        parent_blob = os.path.dirname(self.blob_name.rstrip("/"))
        if parent_blob:
            return GCSPath(f"gs://{self.bucket_name}/{parent_blob}/", self.client)
        else:
            return GCSPath(f"gs://{self.bucket_name}/", self.client)

    def as_posix(self) -> str:
        return self.path

    def stem(self) -> str:
        name = self.name
        return os.path.splitext(name)[0] if "." in name else name

    def suffixes(self) -> list[str]:
        name = self.name
        parts = name.split(".")
        if len(parts) <= 1:
            return []
        return ["." + part for part in parts[1:]]

    def with_name(self, name: str) -> "GCSPath":
        if not self.blob_name:
            raise ValueError("Cannot change name of bucket root")
        parent_path = os.path.dirname(self.blob_name.rstrip("/"))
        if parent_path:
            new_blob = f"{parent_path}/{name}"
        else:
            new_blob = name
        return GCSPath(f"gs://{self.bucket_name}/{new_blob}", self.client)

    def with_suffix(self, suffix: str) -> "GCSPath":
        stem = self.stem()
        return self.with_name(stem + suffix)

    def with_stem(self, stem: str) -> "GCSPath":
        return self.with_name(stem + self.suffix)

    def parts(self) -> tuple[str, ...]:
        parts = ["gs://", self.bucket_name]
        if self.blob_name:
            blob_parts = self.blob_name.strip("/").split("/")
            parts.extend(blob_parts)
        return tuple(parts)

    def relative_to(self, other: "GCSPath") -> "GCSPath":
        if not isinstance(other, GCSPath):
            raise TypeError("Can only compute relative path to another GCSPath")

        if self.bucket_name != other.bucket_name:
            raise ValueError("Cannot compute relative path across different buckets")

        if not other.blob_name:
            # Relative to bucket root
            return GCSPath(f"gs://{self.bucket_name}/{self.blob_name}", self.client)

        self_parts = self.blob_name.strip("/").split("/")
        other_parts = other.blob_name.strip("/").split("/")

        # Find common prefix
        common_len = 0
        for i, (a, b) in enumerate(zip(self_parts, other_parts)):  # noqa:B905
            if a == b:
                common_len = i + 1
            else:
                break

        up_levels = len(other_parts) - common_len
        down_parts = self_parts[common_len:]

        relative_parts = [".."] * up_levels + down_parts
        relative_blob = "/".join(relative_parts) if relative_parts else "."

        return GCSPath(f"gs://{self.bucket_name}/{relative_blob}", self.client)

    def is_absolute(self) -> bool:
        return True

    def resolve(self) -> "GCSPath":
        return self

    def rmdir(self) -> None:
        if not self.is_dir():
            raise NotADirectoryError(f"'{self.path}' is not a directory")

        items = list(self.iterdir())
        if items:
            raise OSError(f"Directory not empty: '{self.path}'")

        if self.blob_name:
            keep_blob = self.bucket.blob(self.blob_name.rstrip("/") + "/.keep")
            if keep_blob.exists():
                keep_blob.delete()

    def unlink(self, missing_ok: bool = False) -> None:
        if not self.blob:
            if missing_ok:
                return
            raise FileNotFoundError(f"'{self.path}' does not exist")

        if not self.blob.exists():
            if missing_ok:
                return
            raise FileNotFoundError(f"'{self.path}' does not exist")

        self.blob.delete()

    def rename(self, target: "GCSPath") -> "GCSPath":
        if not isinstance(target, GCSPath):
            raise TypeError("Target must be a GCSPath")

        if not self.blob:
            raise ValueError("Cannot rename bucket root")

        target.write_bytes(self.read_bytes())

        self.unlink()

        return target

    def stat(self) -> dict[str, Any]:
        if not self.blob:
            raise ValueError("Cannot get stats for bucket root")

        if not self.blob.exists():
            raise FileNotFoundError(f"'{self.path}' does not exist")

        self.blob.reload()
        return {
            "size": self.blob.size or 0,
            "mtime": self.blob.updated.timestamp() if self.blob.updated else 0,
            "ctime": self.blob.time_created.timestamp() if self.blob.time_created else 0,
            "atime": self.blob.updated.timestamp() if self.blob.updated else 0,  # GCS doesn't track access time
            "etag": self.blob.etag,
            "content_type": self.blob.content_type,
            "generation": self.blob.generation,
        }


class PathManager:
    """Factory class for creating appropriate path objects"""

    def __init__(
        self,
        gcs_client: storage.Client | None = None,
        gcs_credentials_path: str | None = None,
    ):
        self._gcs_client = gcs_client
        self._gcs_credentials_path = gcs_credentials_path

    @property
    def gcs_client(self):
        if self._gcs_client is None:
            try:
                if self._gcs_client is None:
                    if self._gcs_credentials_path:
                        from google.oauth2 import service_account

                        credentials = service_account.Credentials.from_service_account_file(self._gcs_credentials_path)
                        self._gcs_client = storage.Client(credentials=credentials)
                    else:
                        self._gcs_client = storage.Client()
            except Exception:
                ...
        return self._gcs_client

    def __call__(self, path: str | Path) -> UniversalPath:
        """Create appropriate path object based on path string"""
        path_str = str(path)
        if path_str.startswith("gs://"):
            return GCSPath(path_str, self.gcs_client)
        else:
            return LocalPath(path_str)


class MLUtilPath(PathManager):
    """Extended path manager with ML-specific utilities"""

    def save_jax_array(self, array: jax.Array, path: str | UniversalPath, format: str = "npy") -> None:  # noqa:A002
        """Save JAX array in various formats"""
        if isinstance(path, str):
            path = self(path)

        if format == "npy":
            buffer = io.BytesIO()
            np.save(buffer, np.array(array))
            path.write_bytes(buffer.getvalue())
        elif format == "pickle":
            buffer = io.BytesIO()
            pickle.dump(array, buffer)
            path.write_bytes(buffer.getvalue())
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_jax_array(self, path: str | UniversalPath, format: str = "npy") -> jax.Array:  # noqa:A002
        """Load JAX array from various formats"""
        if isinstance(path, str):
            path = self(path)

        data = path.read_bytes()
        buffer = io.BytesIO(data)

        if format == "npy":
            return jnp.array(np.load(buffer))
        elif format == "pickle":
            return pickle.load(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_dict(self, data: dict[str, Any], path: str | UniversalPath, format: str = "json") -> None:  # noqa:A002
        """Save dictionary in various formats"""
        if isinstance(path, str):
            path = self(path)

        if format == "json":
            serializable_data = self._make_json_serializable(data)
            path.write_text(json.dumps(serializable_data, indent=2))
        elif format == "pickle":
            buffer = io.BytesIO()
            pickle.dump(data, buffer)
            path.write_bytes(buffer.getvalue())
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_dict(self, path: str | UniversalPath, format: str = "json") -> dict[str, Any]:  # noqa:A002
        """Load dictionary from various formats"""
        if isinstance(path, str):
            path = self(path)

        if format == "json":
            return json.loads(path.read_text())
        elif format == "pickle":
            data = path.read_bytes()
            buffer = io.BytesIO(data)
            return pickle.load(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _make_json_serializable(self, obj):
        """Convert JAX arrays and other non-serializable objects to JSON-safe types"""
        if isinstance(obj, jax.Array | np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer | np.floating):
            return obj.item()
        else:
            return obj

    def copy_tree(self, src: str | UniversalPath, dst: str | UniversalPath) -> None:
        """Copy entire directory tree between local and GCS"""
        if isinstance(src, str):
            src = self(src)
        if isinstance(dst, str):
            dst = self(dst)

        if src.is_file():
            data = src.read_bytes()
            dst.write_bytes(data)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                dst_item = dst / item.name
                self.copy_tree(item, dst_item)


EasyPath: MLUtilPath = MLUtilPath(gcs_credentials_path=os.getenv("EASYDEL_GCS_CLIENT", None))
EasyPathLike: tp.TypeAlias = GCSPath | LocalPath | MLUtilPath
