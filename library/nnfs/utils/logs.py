# Initialize the terminal for color
import logging
import os

LOG_LEVEL=logging.WARNING
print(LOG_LEVEL)

class ColourAndCustomFormatter(logging.Formatter):
    """Adds colours to the log files.

    Returns ANSI in logs.
    https://marketplace.visualstudio.com/items?itemName=iliazeus.vscode-ansi"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(name)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Set up logger as usual
def create_logger(logger_name):
    """Creates a logger.

    Parameters
    ----------
    logger_name : string
        The file directory of the log file

    Returns
    -------
    Logger
    """
    print("Creating loggers")
    logger = logging.getLogger(logger_name)
    try:
        os.mkdir(os.getcwd() + '/program_logs/')
    except FileExistsError:
        pass
    logger_name = os.getcwd() + '/program_logs/'+logger_name+'.log'
    logger.setLevel(LOG_LEVEL)
    shandler = logging.FileHandler(logger_name)
    shandler.setFormatter(ColourAndCustomFormatter())
    logger.addHandler(shandler)
    return logger


def log_IO(name):
    """Decorator function that can be used to log a functions input and output

    Parameters
    ----------
    name : str
        Name of the logger.
    """
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.info("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor

# if __name__=="__main__":
#     # Example usage
#     @log_IO('test')
#     def test_func(key):
#         return key

#     log=create_logger('test')
#     test_func('Input')