import logging
import re
from src.utils.config import LOG_PATH

RESULT_LEVEL = 25
logging.addLevelName(RESULT_LEVEL, 'RESULT')

def result(self, message, *args, **kwargs):
    if self.isEnabledFor(RESULT_LEVEL):
        self._log(RESULT_LEVEL, f'\n\n===== RESULT =====\n{message}\n=================\n', args, kwargs)

logging.Logger.result = result

class PathFilter(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'[A-Za-z]:\\.*?(?=[\\/])', '', str(record.msg))
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.addFilter(PathFilter())
