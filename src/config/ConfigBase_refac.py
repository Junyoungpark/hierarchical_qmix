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
        self._confs = dict()

        for key, arg in kwargs.items():
            if isinstance(arg, dict) or isinstance(arg, ConfigBase):
                self._confs[key] = arg

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            if '_confs' in self.__dict__:
                if key in self.__dict__['_confs']:
                    conf = self.__dict__['_confs'][key]
                    value = self.set_configs(value, conf)

        if isinstance(value, dict) or isinstance(value, ConfigBase):
            if '_confs' in self.__dict__:
                self._confs[key] = value

        self.__dict__[key] = value

    @staticmethod
    def set_configs(target_conf_dict, conf):
        if conf is None:
            return

        for key, val in conf.items():
            if key in target_conf_dict:
                target_conf_dict[key] = val
            else:
                warnings.warn("Unexpected config {} is ignored".format(key))
        return target_conf_dict

    def __getattribute__(self, item):
        item = super().__getattribute__(item)
        if isinstance(item, dict):
            if 'prefix' in item:
                item = self.get_conf(item)
        return item

    @staticmethod
    def get_conf(conf):
        conf = deepcopy(conf)
        _ = conf.pop('prefix')
        return conf

    def __call__(self, pass_arg=None):
        all_confs = dict()
        confs = deepcopy(self.__dict__)
        _ = confs.pop('_confs')
        for _, conf_vals in confs.items():
            if isinstance(conf_vals, dict):
                conf_vals = deepcopy(conf_vals)
                prefix = conf_vals.pop('prefix')
                for conf_k, conf_v in conf_vals.items():
                    if pass_arg is not None:
                        if conf_k in pass_arg:
                            continue
                    conf_name = ' '.join([prefix, conf_k])

                    if isinstance(conf_v, dict):
                        all_confs.update(flatten(conf_v, parent_key=conf_name, sep=" "))
                    if isinstance(conf_v, ConfigBase):
                        all_confs.update(conf_v._get_confs(parent_key=conf_name, sep=" "))

                    else:
                        all_confs[conf_name] = conf_v
            if isinstance(conf_vals, ConfigBase):
                _cfs = conf_vals._get_confs(parent_key=self.name, sep=" ")
                confs.update(_cfs)

        return all_confs

    def _get_confs(self, parent_key, sep=" "):
        confs = {}
        _conf = deepcopy(self.__dict__)
        _ = _conf.pop('_confs')
        for key, value in _conf.items():
            if key == 'name':
                continue

            if isinstance(value, dict) or isinstance(value, ConfigBase):
                if isinstance(value, dict):
                    iterator = value.items()
                if isinstance(value, ConfigBase):
                    iterator = value._get_confs(parent_key=sep.join([parent_key, self.name]), sep=sep)
                for _k, _v in iterator:
                    _k = '_'.join([parent_key, _k])
                    confs[_k] = _v
                continue

            confs[key] = value
        return confs