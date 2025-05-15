import logging

_LOGGER_CONFIGURED = False

def setup_logger(log_file: str, level=logging.DEBUG):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return  

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    formatter = logging.Formatter(
    '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    _LOGGER_CONFIGURED = True


def get_logger(name: str):
    return logging.getLogger(name)
