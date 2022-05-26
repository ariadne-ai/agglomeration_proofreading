__author__ = ['My-Tien Nguyen', 'Norbert Pfeiler', 'Fabian Svara', 'Jörgen Kornfeld']
__copyright__ = "Copyright (C) 2022 ariadne.ai ag"
__license__ = "AGPL v3"
__version__ = "0.1"


import multiprocessing

import concurrent
import logging
import time
import traceback

from flask import Flask, request
import json
from werkzeug.middleware.proxy_fix import ProxyFix

from shared import *
from extract import MeshRetriever


class MeshServer(object):
    def __init__(self, user_db_path='users.txt', mesh_path='file://meshes', lod_list_path='lod.txt'):
        self.user_db_path = user_db_path
        self.graph_lock = multiprocessing.Lock()
        self.multiprocess_manager = multiprocessing.Manager()
        self.graph_semaphore = self.multiprocess_manager.BoundedSemaphore(value=1)
        t = time.time()
        logging.info('Mesh server starting up')
        load_user_db(self)
        self.mesh_retriever = MeshRetriever(mesh_path)
        init_lod_table(self, lod_list_path)
        logging.info(f'Startup took {time.time() - t:.2} s.')

    def get_meshes(self, ssv_ids, lod):
        t = time.time()
        logging.info(f'get_mesh id0: {ssv_ids[0]}, #{len(ssv_ids)}, lod: {lod}')

        if len(ssv_ids) == 0:
            return {'meshes': []}

        recommended_lod = get_recommended_lod(self, ssv_ids)
        if recommended_lod > lod:
            raise UserError(f'This server only serves the recommended level of detail or lower. In this case recommended: {recommended_lod}, requested: {lod}.')
        try:
            meshes = self.mesh_retriever.retrieve_meshes(ssv_ids, lod, timeout=None)
        except (concurrent.futures.TimeoutError, concurrent.futures.process.BrokenProcessPool):
            raise TimeoutError('Request did not finish successfully.')
        logging.info(f'get_meshes took {time.time() - t:.2} s')
        return {'meshes': meshes}


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)
mesh_server = None
wrapper = RequestWrapper()
setup_logging('mesh_logs')


@app.route("/", methods=['GET'])
def route_hello():
    logging.info(f'{request.remote_addr} – {request.url}')
    return json.dumps({'Welcome to': 'Mesh server'})


@app.route('/reload_users')
def reload_user_db():
    logging.info(f'{request.remote_addr} – {request.url}')
    if request.remote_addr == '127.0.0.1':
        load_user_db(mesh_server)
        return ''
    return '', 401


@app.route('/<USER>/<IDENTIFIER>/meshes', methods=['POST'])
@wrapper.auth_and_lock(use_lock=False)
def route_get_meshes(USER, IDENTIFIER):
    try:
        ssv_ids = [int(ssv_id) for ssv_id in request.json['supervoxelIds']]
        lod = request.json.get('lod', 0)
        if not (0 <= lod <= 3):
            raise ValueError(f'Passed lod ({lod}) out of range [0 – 3].')
        if len(ssv_ids) > 100_000:
            logging.warning(f'Requested meshes for {len(ssv_ids)} subobjects, server accepts at most 100k.')
            return f'The server currently only returns meshes for at most 100k subobjects in one request.', 413
    except (KeyError, ValueError):
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"supervoxelIds": [<id>, …, <id>], "lod": 0|1|2|3}, "lod" is optional.', 400
    return json.dumps(mesh_server.get_meshes(ssv_ids, lod))
