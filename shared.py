__author__ = ['My-Tien Nguyen', 'Norbert Pfeiler', 'Fabian Svara', 'Jörgen Kornfeld']
__copyright__ = "Copyright (C) 2022 ariadne.ai ag"
__license__ = "AGPL v3"
__version__ = "0.1"


import functools
import logging
from logging import handlers
import os
import re
import sys
import traceback

from flask import request


from enum import Enum
class Role(Enum):
    ANNOTATOR = 0
    ADMIN = 1
    READONLY = 2

class TimeoutError(Exception):
    pass


class UserError(Exception):
    pass


class RequestWrapper:
    def __init__(self):
        self.server = None

    def auth_and_lock(self, role=Role.READONLY, use_lock=True):
        def wrapper(func):
            @functools.wraps(func)
            def wrapper_(*args, **kwargs):
                logging.info(f'{request.remote_addr} – {request.url}')
                user = kwargs['USER']
                if (role == Role.ADMIN and user not in self.server.admins)\
                    or (role == Role.ANNOTATOR and user in self.server.readonly)\
                    or (user not in self.server.user_ids):
                    return 'You have insufficient privileges for this request.', 401
                lock = self.server.graph_lock if use_lock else self.server.graph_semaphore
                if not lock.acquire(timeout=60):
                    return 'Server is busy. Timeouted after waiting 60 seconds. You can try again.', 408
                try:
                    response = func(**kwargs)
                except UserError as e:
                    logging.error(traceback.format_exc())
                    response = str(e), 400
                except TimeoutError as e:
                    response = str(e), 408
                finally:
                    lock.release()
                return response
            return wrapper_
        return wrapper


def load_user_db(server):
    server.user_ids = []
    server.user_names = []
    server.admins = set()
    server.readonly = set()
    with open(server.user_db_path, 'r') as user_db:
        for line in user_db:
            user_id, username, role = line[:-1].split('\t')
            server.user_ids.append(user_id)
            server.user_names.append(username)
            role = Role(int(role))
            if role == Role.ADMIN:
                server.admins.add(user_id)
            elif role == Role.READONLY:
                server.readonly.add(user_id)


def init_lod_table(server, lod_list_path):
    with open(lod_list_path, 'r') as lod_file:
        server.lod_map = { int(group[1]): int(group[2]) for group in map(lambda line: re.match(r'(\d+)\t(\d+)$', line), lod_file.readlines()) }


def get_recommended_lod(server, ssv_ids):
    return max([0] + [server.lod_map[svid] for svid in set(ssv_ids) & server.lod_map.keys()])


class ColorFormatter(logging.Formatter):
    grey = '\x1b[38;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    format = '%(asctime)s %(levelname)-8s %(message)s (%(filename)s:%(lineno)d)'
    FORMATS = {
        logging.DEBUG: f'{grey}{format}{reset}',
        logging.INFO: f'{grey}{format}{reset}',
        logging.WARNING: f'{yellow}{format}{reset}',
        logging.ERROR: f'{red}{format}{reset}',
        logging.CRITICAL: f'{bold_red}{format}{reset}'
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%dT%H:%M:%S')
        return formatter.format(record)


def setup_logging(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    logfile = logging.handlers.RotatingFileHandler(f'{log_folder}/log.txt', maxBytes=52428800, backupCount=1000) # 50 MiB log files
    console = logging.StreamHandler(stream=sys.stdout)
    formatter = ColorFormatter()
    logfile.setFormatter(formatter)
    console.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logfile, console])
