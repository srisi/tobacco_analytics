

from subprocess import call
import tobacco.utilities.logger
import logging
logger = logging.getLogger('tobacco')

def extract(path):
    logger.info("Extracting {}".format(path))
    call("pigz -d -f {}".format(path), shell=True)
    logger.info("Extraction finished. New file path: {}".format(path[:-3]))
    return path[:-3]    #return new file path (minus '.gz')

def compress(path):
    logger.info("Compressing {}".format(path))
    call("pigz --best -f {}".format(path), shell=True)
    logger.info("Extraction finished. New file path: {}".format("{}.gz".format(path)))
    return "{}.gz".format(path)    #return new file path: added '.gz'