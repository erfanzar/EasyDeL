import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf


def resume_to_file(files_iter, resume_at_file=None):
    _file = next(files_iter)
    while _file != resume_at_file:
        _file = next(files_iter)
    assert _file == resume_at_file
    return files_iter, _file


def list_files(path):
    is_gcs_path = path.startswith('gs://')
    filenames = tf.io.gfile.glob(path) if is_gcs_path else [str(p) for p in Path(path).glob(path)]
    return sorted(filenames)


def create_iterator_from_tfrecords_files(filenames, sequence_length, batch_size, skip=0):
    def parse_fn(sample):
        parsed_features = tf.io.parse_single_example(sample, {'text': tf.io.VarLenFeature(tf.int64)})
        return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

    def truncate_fn(sample):
        return sample[:sequence_length]

    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.repeat()
    ds = ds.skip(skip)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(truncate_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=np.prod(batch_size), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for batch in ds:
        yield batch.numpy().reshape([batch_size[0], batch_size[1], sequence_length])


def create_resumable_iter(files, batch_size, sequence_length, resume_at_file=None, resume_at_batch=None):
    def iterator_from_tfrecords_files(files, sequence_length, batch_size):

        def parse_fn(sample):
            parsed_features = tf.io.parse_single_example(sample, {'text': tf.io.VarLenFeature(tf.int64)})
            return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

        def truncate_fn(sample):
            return sample[:sequence_length]

        ds = tf.data.TFRecordDataset(
            files
        ).map(
            parse_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            truncate_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(
            batch_size=np.prod(batch_size), drop_remainder=True
        ).prefetch(
            tf.data.AUTOTUNE
        )

        for batch in ds:
            yield batch.numpy().reshape([batch_size[0], batch_size[1], sequence_length])

    files_iter = iter(itertools.cycle(files))

    if resume_at_file is not None:
        files_iter, _file = resume_to_file(files_iter, resume_at_file)
        batch_iter = iterator_from_tfrecords_files(_file, sequence_length=sequence_length, batch_size=batch_size)
        for _index, records in enumerate(batch_iter):
            if _index < resume_at_batch:
                continue
            yield records, _file, _index

    for _file in files_iter:
        batch_iter = iterator_from_tfrecords_files(_file, sequence_length=sequence_length, batch_size=batch_size)
        for _index, records in enumerate(batch_iter):
            yield records, _file, _index


def create_mock_iter(vocab_size, batch_size, sequence_length):
    _index = 0
    while True:
        yield np.random.randint(vocab_size, size=[batch_size[0], batch_size[1], sequence_length]), 'file', _index
        _index += 1

