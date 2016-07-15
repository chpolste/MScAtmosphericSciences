"""Metaprogramming for convenient record definition."""

from collections import OrderedDict
from fortranformat import FortranRecordWriter
from fortranformat._exceptions import InvalidFormat

__all__ = ["Record"]


class DummyRecordWriter:
    _eds = ()

    def __call__(self, value):
        return str(value[0])


class ValueDescriptor:

    def __init__(self, name, fmt, default, type_, docstring):
        self.name = name
        if fmt is None: self.fmt = DummyRecordWriter()
        else: self.fmt = FortranRecordWriter(fmt)
        self.type_ = type_
        self.__doc__ = docstring
        # Duplicate None if element is multivalued
        if default is None and isinstance(type_, tuple):
            default = (None,)*len(type_)
        self.default = default

    def __get__(self, instance, owner):
        """Return with formatting applied."""
        if self.name in instance._values:
            value = instance._values[self.name] 
        else:
            value = self.default
        if self.ismultivalue:
            return "".join(map(self._str_value, value))
        return self._str_value(value)

    def __set__(self, instance, value):
        self._check_type(value)
        instance._values[self.name] = value

    def _check_type(self, value):
        """Typecheck for values. Every type accepts None."""
        def isok(v, t):
            return (v is None
                    or (isinstance(t, set) and v in t)
                    or (isinstance(t, range) and t.start <= v < t.stop)
                    or isinstance(v, t)
                    )
        # All types must match for a tuple
        if self.ismultivalue:
            matches = all(isok(v, t) for v, t in zip(value, self.type_))
        else:
            matches = isok(value, self.type_)
        if not matches:
            err = "Value {} is not of type {} or None.".format(
                    value, self.type_)
            raise TypeError(err)

    @property
    def ismultivalue(self):
        return isinstance(self.type_, tuple)

    def _str_value(self, value):
        if value is None:
            return " "*self._format_width()
        return self.fmt.write([value])

    def _format_width(self):
        """With of the Fortran format.

        Necessary for blank fields. Implementation is more of a guess than
        anything but good enough for this application.
        """
        total = 0
        for item in self.fmt._eds:
            repeat = getattr(item, "num_chars", None)
            if repeat is None: repeat = getattr(item, "repeat", 1)
            if repeat is None: repeat = 1
            total = total + repeat*getattr(item, "width", 1)
        return total


class RecordMeta(type):
    """Metaclass for a record definition.
    
    Each element of a record is specified by a 4-tuple:
    [0] Fortran format specifier for output.
    [1] Default value. If None, output will be left blank (blank fields mean
        that MonoRTM picks the default value).
    [2] Type of value, a set of values that are accepted, a range object
        specifying accepted values or a tuple for multivalue elements.
    [3] Docstring.
    """

    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict() # Order of elements in record is important

    def __new__(cls, name, bases, dct):
        namespace = {}
        namespace["_order"] = []
        for key, value in dct.items():
            if callable(value) or key.startswith("_"):
                # Exception for necessary class attributes and methods
                namespace[key] = value
                continue
            namespace["_order"].append(key)
            namespace
            namespace[key] = ValueDescriptor(key, *value)
        return super().__new__(cls, name, bases, namespace)

