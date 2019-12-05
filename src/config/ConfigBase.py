from copy import deepcopy
import warnings
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

    def __init__(self, **kwargs):
        self._confs = dict()
        for key, arg in kwargs.items():
            if isinstance(arg, dict):
                self._confs[key] = arg

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            if '_confs' in self.__dict__:
                if key in self.__dict__['_confs']:
                    conf = self.__dict__['_confs'][key]
                    value = value
                    value = self.set_configs(value, conf)
        self.__dict__[key] = value

    def __getattribute__(self, item):
        item = super().__getattribute__(item)
        if isinstance(item, dict):
            if 'prefix' in item:
                item = self.get_conf(item)
        return item

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
                    else:
                        all_confs[conf_name] = conf_v
        return all_confs


class ConfigDict:

    def __init__(self, name):
        super(ConfigDict, self).__init__()
        self._config_dicts = {}
        self.name = name

    def __setattr__(self, key, value):
        if isinstance(value, ConfigDict):
            self._config_dicts[key] = value

        self.__dict__[key] = value

    def __call__(self):
        confs = {}
        _conf = deepcopy(self.__dict__)
        _ = _conf.pop('_config_dicts')
        for key, value in _conf.items():
            if key == 'name':
                continue

            if isinstance(value, dict) or isinstance(value, ConfigDict):
                if isinstance(value, dict):
                    iterator = value.items()
                if isinstance(value, ConfigDict):
                    iterator = value().items()

                for _k, _v in iterator:
                    _k = '_'.join([self.name, _k])
                    confs[_k] = _v
                continue

            confs[key] = value
        return confs


if __name__ == "__main__":
    dict_a = ConfigDict('A')
    dict_b = ConfigDict('B')
    dict_b.conf_b1 = {'conf_b1_1': 1}

    dict_c = ConfigDict('C')
    dict_c.conf_c1 = {'conf_c1_1': 2}

    dict_b.testtest = dict_c
    dict_a.test = dict_b
    print(dict_a())

    print("test")
