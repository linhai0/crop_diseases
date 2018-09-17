import logging


def get_logger(config):
    logger = logging.getLoggerClass(config.logger_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
    file_handler = logging.FileHandler(config.logger_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def batch_log(logger, log_interval, batch_idx, batch_size, loss):
    if batch_idx and batch_idx % log_interval:
        logger.info('[{0} / {1}] batch {2}, loss {3}'.format(
            batch_idx // log_interval,
            batch_size // log_interval,
            batch_idx,
            round(loss, 3)))


def sample_log(logger, log_interval, sample_idx, sample_size):
    if sample_idx and sample_idx % log_interval:
        logger.info(
            '[{0} / {1}] sample {2}'.format(sample_idx // log_interval, sample_size // log_interval, sample_idx))


if __name__ == '__main__':
    pass