import importlib

class ModuleImporter:
    """ This class allows us to Lazily import ROS modules to make execution in non-ROS workspaces possible.
        Additionally caches the imports to support real-time publishing when using RosPublisher.py """
    
    _module_cache: dict = {}
    _attr_cache: dict = {}

    @classmethod
    def get_module_attribute(cls, module_path: str, attr_name: str):
        """ Dynamically loads and caches a ROS attribute (like a Class). """

        key = f"{module_path}.{attr_name}"
        if key not in cls._attr_cache:
            module = cls.get_module(module_path)
            try:
                cls._attr_cache[key] = getattr(module, attr_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Module '{module_path}' was found, but attribute '{attr_name}' does not exist. Ensure you have the correct ROS message packages installed.") from e
        return cls._attr_cache[key]

    @classmethod
    def get_module(cls, name: str):
        """ Returns a cached reference to a top-level module or sub-module (e.g., 'rospy' or 'sensor_msgs.msg'). """

        if name not in cls._module_cache:
            try:
                cls._module_cache[name] = importlib.import_module(name)
            except ImportError as e:
                raise ImportError(f"Module {name} is required but not installed.") from e
        return cls._module_cache[name]

    