"""Compatibilidad: tkinter.tix fue eliminado en Python 3.13.

tkcap 0.0.4 todavia lo importa. Este modulo registra un tkinter.tix
sintetico que delega en tkinter, y debe importarse ANTES que tkcap.
"""
import sys
import tkinter
import types

if "tkinter.tix" not in sys.modules:
    _tix = types.ModuleType("tkinter.tix")
    _tix.__getattr__ = lambda name: getattr(tkinter, name)
    sys.modules["tkinter.tix"] = _tix
    tkinter.tix = _tix
