__version__ = "0.3.0"


from importlib_resources import read_text

import aspyre
from aspyre.utils.config import Config


config = Config(read_text(aspyre, 'config.json'))
