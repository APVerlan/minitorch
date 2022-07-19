from typing import Dict, Any, Optional


class Module:
    ...


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, value: Any = None, name: Optional[str] = None) -> None:
        self.value: Any = value
        self.name: Optional[str] = name

        if hasattr(value, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, value: Any) -> None:
        "Update the parameter value."
        self.value: Any = value

        if hasattr(value, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode

    """

    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Module] = {}
        self.training: bool = True

    def modules(self) -> list[Module]:
        "Return the direct child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self) -> None:
        "Set the mode of this module and all descendent modules to `train`."
        self.training = True
        for mod in self.modules():
            mod.train()

    def eval(self) -> None:
        "Set the mode of this module and all descendent modules to `eval`."
        self.training = False
        for mod in self.modules():
            mod.eval()

    def named_parameters(self) -> list[tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        res = list(self.__dict__["_parameters"].items())

        for mod_name, mod in self.__dict__["_modules"].items():
            res += [
                (mod_name + "." + item[0], item[1]) for item in mod.named_parameters()
            ]
        return res

    def parameters(self) -> list[Parameter]:
        "Enumerate over all the parameters of this module and its descendents."
        res = list(self.__dict__["_parameters"].values())

        for module in self.modules():
            res += module.parameters()
        return res

    def add_parameter(self, name: str, value: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            name (str): Local name of the parameter.
            value (Any): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        param = Parameter(value, name)
        self.__dict__["_parameters"][name] = param
        return param

    def __setattr__(self, name: str, val: Any) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][name] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][name] = val
        else:
            super().__setattr__(name, val)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][name]

        if name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("")

    def __repr__(self) -> str:
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
