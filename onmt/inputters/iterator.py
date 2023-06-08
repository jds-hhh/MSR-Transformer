"""Contains all methods relate to iteration."""
import torchtext.data

from onmt.utils.logging import logger


def batch_iter(data, batch_size, batch_size_fn=None, batch_size_multiple=1):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if batch_size_multiple > 1:
                overflowed += (
                    (len(minibatch) - overflowed) % batch_size_multiple)
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                if overflowed == len(minibatch):
                    logger.warning(
                        "The batch will be filled until we reach %d,"
                        "its size may exceed %d tokens"
                        % (batch_size_multiple, batch_size)
                        )
                else:
                    yield minibatch[:-overflowed]
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1, size_so_far)
    if minibatch:
        yield minibatch


def _pool(data, batch_size, batch_size_fn, batch_size_multiple,
          sort_key, random_shuffler, pool_factor):
    for p in torchtext.data.batch(
            data, batch_size * pool_factor,
            batch_size_fn=batch_size_fn):
        p_batch = list(batch_iter(
            sorted(p, key=sort_key),
            batch_size,
            batch_size_fn=batch_size_fn,
            batch_size_multiple=batch_size_multiple))
        for b in random_shuffler(p_batch):
            yield b



class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Deprecated: this attribute is left for backwards compatibility,
            however it is UNUSED as of the merger with pytorch 0.4.
        input_fields: The names of the fields that are used as input for the model
        target_fields: The names of the fields that are used as targets during
                       model training

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=None, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.fields = dataset.fields.keys()
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.__dict__:
            return 'Empty {} instance'.format(torch.typename(self))

        fields_to_index = filter(lambda field: field is not None, self.fields)
        var_strs = '\n'.join(['\t[.' + name + ']' + ":" + _short_str(getattr(self, name))
                              for name in fields_to_index if hasattr(self, name)])

        data_str = (' from {}'.format(self.dataset.name.upper())
                    if hasattr(self.dataset, 'name')
                    and isinstance(self.dataset.name, str) else '')

        strt = '[{} of size {}{}]\n{}'.format(torch.typename(self),
                                              self.batch_size, data_str, var_strs)
        return '\n' + strt

    def __len__(self):
        return self.batch_size

    def _get_field_values(self, fields):
        if len(fields) == 0:
            return None
        elif len(fields) == 1:
            return getattr(self, fields[0])
        else:
            return tuple(getattr(self, f) for f in fields)

    def __iter__(self):
        yield self._get_field_values(self.input_fields)
        yield self._get_field_values(self.target_fields)


class OrderedIterator(torchtext.data.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_factor=1,
                 batch_size_multiple=1,
                 yield_raw_example=False,
                 **kwargs):
        super(OrderedIterator, self).__init__(dataset, batch_size, **kwargs)
        self.batch_size_multiple = batch_size_multiple
        self.yield_raw_example = yield_raw_example
        self.dataset = dataset
        self.pool_factor = pool_factor

    def create_batches(self):
        if self.train:
            if self.yield_raw_example:
                self.batches = batch_iter(
                    self.data(),
                    1,
                    batch_size_fn=None,
                    batch_size_multiple=1)
            else:
                self.batches = _pool(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor)
        else:
            self.batches = []
            for b in batch_iter(
                    self.data(),
                    self.batch_size,
                    batch_size_fn=self.batch_size_fn,
                    batch_size_multiple=self.batch_size_multiple):
                self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        """
        Extended version of the definition in torchtext.data.Iterator.
        Added yield_raw_example behaviour to yield a torchtext.data.Example
        instead of a torchtext.data.Batch object.
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    #  requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                if self.yield_raw_example:
                    yield minibatch[0]
                else:
                    yield torchtext.data.Batch(
                        minibatch,
                        self.dataset,
                        self.device)
            if not self.repeat:
                return


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
