import matplotlib.pyplot as plt
import matplotlib
import logging, os

from sklearn.metrics import confusion_matrix
import seaborn as sns
def historm(x):
    sns.distplot(x)#, kde=False, rug=True)
    plt.show()

def sns_confusion_matrix(x,y):
    mat = confusion_matrix(x, y)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=x.target_names,
            yticklabels=x.target_names)
    plt.ylabel('predicted label')




def get_logger(logger_name, logger_dir):
    logger_filepath = os.path.join(logger_dir, logger_name+'.log')
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
    file_handler = logging.FileHandler(logger_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def batch_log(logger, log_interval, batch_idx, batch_num, loss):
    if batch_idx and batch_idx % log_interval == 0:
        logger.info('[{0} / {1}] batch {2}, loss {3}'.format(
            batch_idx // log_interval,
            batch_num // log_interval,
            batch_idx,
            round(loss, 3)))



def sample_log(logger, log_interval, sample_idx, sample_num):
    if sample_idx and sample_idx % log_interval == 0:
        logger.info('[{0} / {1}] sample {2}'.format(
            sample_idx // log_interval,
            sample_num // log_interval,
            sample_idx))
