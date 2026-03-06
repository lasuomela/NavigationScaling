"""Registry to map e.g. model names to corresponding python class."""

from typing import Any, Callable, DefaultDict, Optional, Type, Dict

import collections
from omegaconf import DictConfig

class Singleton(type):
    """
    This metatclass creates Singleton objects by ensuring only one instance is created
    and any call is directed to that instance. The mro() function and following dunders,
    EXCEPT __call__, are inherited from the the stdlib Python library,
    which defines the "type" class.
    """
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)
    
    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a model to registry with key :p:`name`

        :param name: Key with which the measure will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "model", to_register, name, assert_type=None
        )
    
    @classmethod
    def get_model(cls, model_config: DictConfig) -> Type[Any]:
        model_cls = cls._get_impl("model", model_config.type)
        return model_cls
    
registry = Registry()