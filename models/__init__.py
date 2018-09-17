from models.basic_model import BasicAgent
from models.simple_net import sim_net

__all__ = [
    "BasicAgent",
    "sim_net"
]

def make_models(config):
    if config.model_name in __all__:
        return globals()[config.model_name](config)
    else:
        raise Exception('The model name %s does not exist' % config.model_name)

def get_model_class(config):
    if config.model_name in __all__:
        return globals()[config.model_name]
    else:
        raise Exception('The model name %s does not exist' % config.model_name)