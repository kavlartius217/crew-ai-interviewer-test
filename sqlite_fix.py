import sys
import platform

# Only apply the SQLite fix if we're not on Linux
if platform.system() != 'Linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
