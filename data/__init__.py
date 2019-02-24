from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [
    basename(f)[:-3]
    for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]
for module in __all__:
    __import__("data.{}".format(module), locals(), globals())
del module

