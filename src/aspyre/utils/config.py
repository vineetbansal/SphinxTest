"""
Class and utility functions for a Config object driven from a .json file,
with functionality to override values in blocks of code / scripts
"""

import functools
from contextlib import contextmanager
import logging.config
import logging
import json
from copy import deepcopy
from types import SimpleNamespace
from argparse import ArgumentParser


def _rsetattr(obj, attr, val):
    # Recursive setattr
    pre, _, post = attr.rpartition('.')
    try:
        int(post)
    except ValueError:
        return setattr(_rgetattr(obj, pre) if pre else obj, post, val)
    else:
        _rgetattr(obj, pre)[int(post)] = val


def _rgetattr(obj, attr, *args):
    # Recursive getattr
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


@contextmanager
def config_override(config, args):
    try:
        original_namespace = deepcopy(config.namespace)
        for k, v in args.__dict__.items():
            if k.startswith('config.'):
                _rsetattr(config.namespace, k[7:], v)
        yield args
    finally:
        config.namespace = original_namespace


class Config:
    def __init__(self, json_string):
        d = json.loads(json_string)

        # The logging module supports configuration from a dictionary using dictConfig, but not a SimpleNamespace,
        # so take care of that first
        if 'logging' in d:
            logging.config.dictConfig(d['logging'])
        else:
            logging.basicConfig(level=logging.INFO)

        # Now that logging is configured, reload the json, but now with an object hook
        # so we have cleaner access to keys by way of (recursive) attributes
        self.namespace = json.loads(json_string, object_hook=lambda d: SimpleNamespace(**d))

        # Finally, there's no need preserving the 'logging' attribute in this object since that's
        # a one-time configuration and would only interfere with intended use-cases of this object
        if 'logging' in d:
            delattr(self.namespace, 'logging')

    def __getattr__(self, item):
        return getattr(self.namespace, item)

    def flatten(self):
        # Flatten object and return a dictionary of key=>value pairs
        # where key names are delimited using the '.' delimited
        # Adapted from
        #   https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
        out = {}

        def _flatten(x, name=''):
            if type(x) is SimpleNamespace:
                for a in x.__dict__:
                    _flatten(getattr(x, a), name + a + '.')
            elif type(x) is list:
                i = 0
                for a in x:
                    _flatten(a, name + str(i) + '.')
                    i += 1
            else:
                out[name[:-1]] = x

        _flatten(self.namespace)
        return out


class ConfigArgumentParser(ArgumentParser):
    """
    An ArgumentParser that adds arguments found in the (flat) 'config' (of type Config) object used in it's
    constructor. By default, the aspyre.config Config object is used.
    All arguments to the parser are added with the 'config.' prefix
    """

    def __init__(self, *args, **kwargs):
        if 'config' in kwargs:
            self._config = kwargs['config']
            kwargs.pop('config')
        else:
            from aspyre import config
            self._config = config

        super().__init__(*args, **kwargs)

        config_group = self.add_argument_group('config')
        for k, v in self._config.flatten().items():
            config_group.add_argument(f'--config.{k}', default=v, type=type(v))

    def parse_args(self, *args, **kwargs):
        """
        A context manager that parses command line arguments,
        tweaks the Config object associated with this ArgumentParser within the 'with' block,
        and reverts it back to it's original values once the block exits.
        """
        args = super().parse_args(*args, *kwargs)
        return config_override(self._config, args)
