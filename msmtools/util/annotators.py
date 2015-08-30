
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""
from __future__ import absolute_import

import warnings
from functools import wraps
import inspect

__all__ = ['doc_inherit',
           'deprecated',
           'shortcut',
           ]


class DocInherit(object):

    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit


def deprecated(msg):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    msg : str
        a user level hint which should indicate which feature to use otherwise.

    """
    def deprecated_decorator(func):

        def new_func(*args, **kwargs):
            _, filename, line_number, _, _, _ = \
                inspect.getouterframes(inspect.currentframe())[1]

            user_msg = "Call to deprecated function %s. Called from %s line %i. " \
                % (func.__name__, filename, line_number)
            if msg:
                user_msg += msg

            warnings.warn_explicit(
                user_msg,
                category=DeprecationWarning,
                filename=func.__code__.co_filename,
                lineno=func.__code__.co_firstlineno + 1
            )
            return func(*args, **kwargs)

        new_func.__dict__['__deprecated__'] = True

        # TODO: search docstring for notes section and append deprecation notice (with msg)

        return new_func

    return deprecated_decorator


def shortcut(name):
    """Add an shortcut (alias) to a decorated function.

    Calling the shortcut (alias) will call the decorated function. The shortcut name will be appended
    to the module's __all__ variable and the shortcut function will inherit the function's docstring

    Examples
    --------
    In some module you have defined a function
    >>> @shortcut('is_tmatrix') # doctest: +SKIP
    >>> def is_transition_matrix(args): # doctest: +SKIP
    >>>     pass # doctest: +SKIP
    Now you are able to call the function under its short name
    >>> is_tmatrix(args) # doctest: +SKIP

    """
    # TODO: this does not work (is not tested with class member functions)
    # extract callers frame
    frame = inspect.stack()[1][0]
    # get caller module of decorator

    def wrap(f):
        # docstrings are also being copied
        frame.f_globals[name] = f
        if '__all__' in frame.f_globals:
            # add shortcut if it's not already there.
            if name not in frame.f_globals['__all__']:
                frame.f_globals['__all__'].append(name)
        return f
    return wrap