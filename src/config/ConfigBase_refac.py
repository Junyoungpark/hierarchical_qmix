import warnings
from copy import deepcopy
import collections


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ConfigBase:

    def __init__(self, name: str, **kwargs):
        self.name = name  # will be used as the prefix

    def __call__(self, pass_arg=None, base_prefix=None):
        if base_prefix is None:
            base_prefix = self.name

        all_confs = dict()
        confs = deepcopy(self.__dict__)
        for conf_key, conf_vals in confs.items():
            if isinstance(conf_vals, dict):
                conf_vals = deepcopy(conf_vals)
                prefix = conf_key
                for conf_k, conf_v in conf_vals.items():
                    if pass_arg is not None:
                        if conf_k in pass_arg:
                            continue
                    conf_name = ' '.join([base_prefix, prefix, conf_k])

                    if isinstance(conf_v, dict):
                        all_confs.update(flatten(conf_v, parent_key=conf_name, sep=" "))

                    if isinstance(conf_v, ConfigBase):
                        all_confs.update(conf_v(base_prefix=conf_name))

                    all_confs[conf_name] = conf_v

            if isinstance(conf_vals, ConfigBase):
                _cfs = conf_vals(base_prefix=' '.join([self.name, conf_vals.name]))
                all_confs.update(_cfs)

        return all_confs