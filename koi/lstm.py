"""Minimal koi.lstm stub.

Bonito imports this module at top level of bonito/crf/model.py, but the
actual ``koi.lstm.update_graph`` function is only invoked from
``Model.use_koi()``, which is a no-op in our setup (we use the standard
PyTorch encoder path). So a stub that raises NotImplementedError on call
but imports successfully is sufficient.
"""


def update_graph(*args, **kwargs):
    raise NotImplementedError(
        "koi.lstm.update_graph is not available — pure-PyTorch koi stub. "
        "Do not call Model.use_koi(); use the standard encoder path."
    )
