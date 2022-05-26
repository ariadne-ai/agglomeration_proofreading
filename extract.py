__author__ = ['My-Tien Nguyen', 'Norbert Pfeiler', 'Fabian Svara', 'Jörgen Kornfeld']
__copyright__ = "Copyright (C) 2022 ariadne.ai ag"
__license__ = "AGPL v3"
__version__ = "0.1"


import base64
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
from itertools import islice
import os
import psutil
from tqdm import tqdm
from time import time
import traceback

from cloudvolume import CloudVolume

if 'mesh_path' in os.environ:
    print('startup', __name__)
    vol = CloudVolume(os.environ['mesh_path'], parallel=True, mesh_dir='mesh')

def _retrieve_b64enchoded_plys(ssv_ids, lod):
    return {ssv_id: base64.b64encode(mesh.to_ply()).decode('ascii') for ssv_id, mesh in vol.mesh.get(ssv_ids, lod=lod).items()}


import numpy as np
class MeshRetriever():
    def __init__(self, mesh_path):
        os.environ['mesh_path'] = mesh_path
        self.executor = ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'))

    def retrieve_meshes(self, ssv_ids, lod, timeout):
        futures = []
        step = 1000

        t = time()
        try:
            for i in range(0, len(ssv_ids), step):
                futures.append(self.executor.submit(_retrieve_b64enchoded_plys, ssv_ids[i:min(i+step, len(ssv_ids))], lod))
            mesh_dict = {}

            for future in tqdm(as_completed(futures, timeout=timeout), total=len(futures)):
                mesh_dict.update(future.result())
        except (concurrent.futures.TimeoutError, concurrent.futures.process.BrokenProcessPool) as e:
            logging.error(f'Restarting process pool. traceback:\n{traceback.format_exc()}')
            for process in map(psutil.Process, self.executor._processes):
                for child in process.children(recursive=True):
                    child.terminate()
                process.terminate()
            self.executor.shutdown()
            self.executor = ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'))
            raise e
        return [mesh_dict[ssv_id] for ssv_id in ssv_ids]
