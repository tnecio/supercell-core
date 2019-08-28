"""
Contains custom Exceptions, Warnings, and their descriptions
"""

# warn is used by other files importing * from errors
import warnings
from enum import Enum


class LinearDependenceError(Exception):
    """
    An exception thrown when a set of linearly independent vectors is expected,
    but supplied values have a pair of linearly dependent vectors in them.
    """
    pass


class BadUnitError(Exception):
    """
    An exception thrown when specified unit can not be used in a given context
    """
    pass


class UndefinedBehaviourError(Exception):
    """
    An exception thrown when it's not clear what the user's intention was
    """
    pass


class ParseError(Exception):
    """
    An exception thrown when incorrect data are passed to parsing or reading
    functions
    """
    pass


class WarningText(Enum):
    """
    Enumeration that contains warning messages text

    Warnings are logged when some function in the package was used in a way
    that suggests something is incorrect, but it's not clear, and calculation
    can be done anyway
    """
    ZComponentWhenNoZVector = "A z-component of a vector was specified, but only 2 vectors were given, causing the 3rd one to be set to default value"
    ReassigningPartOfVector = "You reassign only a part of a vector, other elements of which were previously set to non-default values; Values of those other elements will be retained"
    AtomOutsideElementaryCell = "Atom position specified outside the elementary cell"
    UnknownChemicalElement = "Chemical element symbol not found in the periodic table"
