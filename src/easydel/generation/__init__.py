# some parts of This section of EasyDeL (generation)
# copied from transformers
# Reason: Transformers is not compatible with TPUs and EasyDeL in same scope due to some EasyDeL natuers
# There's many issues reporting EasyDeL import crash (like https://github.com/erfanzar/EasyDeL/issues/166) and
# more than 80% of these runtime crashes are made by `transformers` and `datasets`
