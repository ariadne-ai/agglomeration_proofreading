__author__ = ['My-Tien Nguyen', 'Norbert Pfeiler', 'Fabian Svara', 'Jörgen Kornfeld']
__copyright__ = "Copyright (C) 2022 ariadne.ai ag"
__license__ = "AGPL v3"
__version__ = "0.1"


import multiprocessing

import logging
import os
from pathlib import Path
import re
import time
import traceback

from shared import *
from flask import Flask, request, send_file, Response
from graph_tool import search
from graph_tool.all import Graph
import json
import numpy as np
from werkzeug.middleware.proxy_fix import ProxyFix


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)
server_state = None
wrapper = RequestWrapper()
setup_logging('logs')

@app.route("/", methods=['GET'])
def route_hello():
    logging.info(f'{request.remote_addr} – {request.url}')
    return json.dumps({'Welcome to': 'Proofreading server'})


@app.route('/reload_users')
def reload_user_db():
    logging.info(f'{request.remote_addr} – {request.url}')
    if request.remote_addr == '127.0.0.1':
        load_user_db(server_state)
        return ''
    return '', 401


class InvalidSegIdError(Exception):
    pass

class DFSTimeout(Exception):
    pass

@app.errorhandler(InvalidSegIdError)
def handle_invalid_seg_id(error):
    return str(error), 400

@app.route('/<USER>/<IDENTIFIER>/mergelist/<SSV_ID>', methods=['GET'])
@wrapper.auth_and_lock()
def route_get_mergelist(USER, IDENTIFIER, SSV_ID):
    whole_cc = request.args.get('cc', 'true').lower() == 'true'
    try:
        ssv_id = int(SSV_ID)
    except:
        logging.error(traceback.format_exc())
        return '', 404
    return json.dumps(server_state.get_mergelist(ssv_id, whole_cc))


@app.route('/<USER>/<IDENTIFIER>/graph/<SSV_ID>', methods=['POST'])
@wrapper.auth_and_lock()
def route_get_graph(USER, IDENTIFIER, SSV_ID):
    try:
        ssv_id = int(SSV_ID)
    except:
        logging.error(traceback.format_exc())
        return '', 400
    return json.dumps(server_state.get_graph(ssv_id))


@app.route('/<USER>/changes/<VOLUME_ID>/<CHANGE_STACK>/equivalences:list', methods=['POST'])
@wrapper.auth_and_lock()
def get_neighbors(USER, VOLUME_ID, CHANGE_STACK):
    try:
        segment_ids = [int(seg_id) for seg_id in request.json['segmentId']]
    except:
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"segmentId": ["<id>"]}', 400
    return json.dumps(server_state.get_neighbors(segment_ids))


@app.route('/<USER>/changes/<VOLUME_ID>/<CHANGE_STACK>/equivalences:getgroups', methods=['POST'])
@wrapper.auth_and_lock()
def route_get_cc(USER, VOLUME_ID, CHANGE_STACK):
    try:
        segment_ids = [int(seg_id) for seg_id in request.json['segmentId']]
    except:
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"segmentId": ["<id>"]}', 400
    return json.dumps(server_state.get_cc(segment_ids))


@app.route('/<USER>/objects/<VOLUME_ID>/meshes/<MESH_KEY>:listfragments')
@wrapper.auth_and_lock()
def route_get_fragments(USER, VOLUME_ID, MESH_KEY):
    if server_state.mesh_path is not None:
        return '', 404
    try:
        object_id = int(request.args['objectId'])
        returnSupervoxels = request.args['returnSupervoxelIds'] == 'true'
        changestack = request.args.get('header.changeStackId')
    except KeyError as e:
        return 'Expected following parameters: objectId, returnSupervoxelIds and optionally header.changeStackId', 400
    return json.dumps(server_state.get_fragments(object_id, changestack is not None))


@app.route('/<USER>/changes/<VOLUME_ID>/<CHANGE_STACK>/equivalences:set', methods=['POST'])
@wrapper.auth_and_lock(role=Role.ANNOTATOR)
def route_add_edge(USER, VOLUME_ID, CHANGE_STACK):
    try:
        edge_first = request.json['edge']['first']
        edge_second = request.json['edge']['second']
    except:
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"edge": {"first": <id>, "second": <id>}}', 400
    return '' if server_state.add_edge(str(USER), edge_first, edge_second) else ('Edge already exists', 400)


@app.route('/<USER>/changes/<VOLUME_ID>/<CHANGE_STACK>/equivalences:delete', methods=['POST'])
@wrapper.auth_and_lock(role=Role.ANNOTATOR)
def route_del_edge(USER, VOLUME_ID, CHANGE_STACK):
    try:
        edge_first = request.json['edge']['first']
        edge_second = request.json['edge']['second']
    except:
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"edge": {"first": <id>, "second": <id>}}', 400
    return '' if server_state.del_edge(str(USER),edge_first, edge_second) else ('Edge doesn’t exist', 400)


@app.route('/<USER>/changes/<VOLUME_ID>/anchors:addnote', methods=['POST'])
@wrapper.auth_and_lock(role=Role.ANNOTATOR)
def route_add_anchor_note(USER, VOLUME_ID):
    try:
        anchor_id = int(request.args['anchorId'])
        note = request.json['note']
    except:
        logging.error(traceback.format_exc())
        return 'Expected parameter anchorId and json body of the form {"note": <note>}', 400
    return '' if server_state.add_anchor_note(str(USER), anchor_id, note) else (f'Anchor {anchor_id} doesn’t exist', 400)


@app.route('/<USER>/changes/<VOLUME_ID>/anchors:add', methods=['POST'])
@wrapper.auth_and_lock(role=Role.ADMIN)
def route_add_anchor(USER, VOLUME_ID):
    try:
        anchor_id = request.json['anchorId']
    except:
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"anchorId": <id>}', 400
    return '' if server_state.add_anchor(str(USER), anchor_id) else (f'Anchor {anchor_id} already exists', 400)


@app.route('/<USER>/changes/<VOLUME_ID>/anchors:delete', methods=['POST'])
@wrapper.auth_and_lock(role=Role.ADMIN)
def route_delete_anchor(USER, VOLUME_ID):
    try:
        anchor_id = int(request.json['anchorId'])
    except:
        logging.error(traceback.format_exc())
        return 'Expected json body of the form {"anchorId": <id>}', 400
    return '' if server_state.delete_anchor(str(USER), anchor_id) else (f'Anchor {anchor_id} doesn’t exist', 400)


from numba import njit
from numba.core import types
from numba.typed import Dict

@njit
def init_id_map(vertices):
    id_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    for v_idx, seg_id in vertices:
        id_map[seg_id] = v_idx
    return id_map

class AgglomerationServer(object):
    def __init__(self, action_db_path='action_db.tsv', anchor_path='anchors.txt', timestamp_path='timestamps.json', user_db_path='users.txt', ssv_edge_list_path='test_graph.gt', dataset_path='confs', lod_list_path='lod.txt'):
        self.action_db_path = Path().cwd() / action_db_path
        self.action_db_path_anonymized = self.action_db_path.parent / 'static-files' / f'changes.tsv'
        self.anchor_path = Path().cwd() / anchor_path
        self.user_db_path = Path().cwd() / user_db_path
        self.ssv_edge_list_path = Path().cwd() / ssv_edge_list_path
        self.timestamp_path = Path().cwd() / timestamp_path
        self.dataset_path = Path().cwd() / dataset_path
        self.graph_lock = multiprocessing.Lock()
        self.multiprocess_manager = multiprocessing.Manager()
        self.graph_semaphore = self.multiprocess_manager.BoundedSemaphore(value=1)
        t = time.time()
        logging.info('Proofreading server starting up')
        load_user_db(self)
        self.init_anchors()
        self.init_timestamps()
        self.init_graph_tool_dict()
        self.init_action_db()
        init_lod_table(self, lod_list_path)
        logging.info(f'Startup took {time.time() - t:.2} s.')


    def vertex_idx(self, seg_id):
        try:
            return self.id_map[seg_id]
        except KeyError:
            raise InvalidSegIdError(f'segment ID {seg_id} does not exist')

    def init_graph_tool_dict(self):
        t = time.time()
        self.ssv_graph = Graph()
        self.ssv_graph.load(str(self.ssv_edge_list_path))
        t2 = time.time()
        logging.info(f'graph loaded in {t2 - t} s')
        self.id_map = init_id_map(self.ssv_graph.get_vertices([self.ssv_graph.vp['seg_ids']]))
        logging.info(f'id map generated in {time.time() - t2} s')

    def init_action_db(self):
        # format:
        # user,ID1,ID2,-\n -> delete edge between ID1 and ID2
        # user,ID1,ID2,+\n -> add edge between ID1 and ID2
        t = time.time()
        try:
            with open(self.action_db_path, 'r') as f, open(self.action_db_path_anonymized, 'w') as f_anon:
                for line_number, action_record in enumerate(f):
                    match = re.match(r'(\w+)\t([^\t]+)\t(.+)', action_record.rstrip('\n'))
                    user, action, params = match.groups()
                    f_anon.write(f'{action}\t{params}\n')
                    if action in {'-', '+'}:
                        ID1, ID2 = (int(ID) for ID in params.split('\t'))
                        if action == '-':
                            edge = self.ssv_graph.edge(self.vertex_idx(ID1), self.vertex_idx(ID2))
                            self.ssv_graph.remove_edge(edge)
                        elif action == '+':
                            self.ssv_graph.add_edge(self.vertex_idx(ID1), self.vertex_idx(ID2))
                    elif action == 'note':
                        match = re.match(r'(\d+)\t(.*)', params)
                        anchor_id, note = match.groups()
                        self.anchors[int(anchor_id)]['note'] = note
                    elif action == 'anchor_add':
                        anchor_id = int(params)
                        if anchor_id in self.anchors:
                            raise ValueError(f'action db line {line_number}: anchor_add for existing anchor {anchor_id}')
                        self.anchors[anchor_id]= {}
                    elif action == 'anchor_del':
                        anchor_id = int(params)
                        exists = anchor_id not in self.anchors
                        mapped = 'nuc' in self.anchors[anchor_id]
                        if anchor_id not in self.anchors or 'nuc' in self.anchors[anchor_id]:
                            raise ValueError(f'action db line {line_number}: invalid anchor_del for {anchor_id}. exists: {exists}. mapped: {mapped}')
                        del self.anchors[anchor_id]
        except FileNotFoundError:
            with open(self.action_db_path, 'a'):
                os.utime(self.action_db_path, None)
        logging.info(f'init action db in {time.time() - t} s')

    def init_anchors(self):
        self.anchors = {}
        with open(self.anchor_path, 'r') as anchors_file:
            for line in anchors_file:
                anchor, nuc, x, y, z = line.rstrip('\n').split('\t')
                self.anchors[int(anchor)] = {'nuc': int(nuc), "pos": (int(x), int(y), int(z))}

    def init_timestamps(self):
        self.timestamps = {}
        if os.path.exists(self.timestamp_path):
            with open(self.timestamp_path, 'r') as timestamp_file:
                self.timestamps = json.load(timestamp_file)
                self.timestamps = {int(key): val for key, val in self.timestamps.items()}

    def record_action(self, user_id, t, action, params):
        params = '\t'.join((f'{param}' for param in params))
        action_str = f'{user_id}\t{action}\t{params}\n'
        with open(self.action_db_path, 'a') as f:
            f.write(action_str)
            f.flush()
            os.fsync(f.fileno())
        with open(self.action_db_path_anonymized, 'a') as f:
            action_str = f'{action}\t{params}\n'
            f.write(action_str)
            f.flush()
            os.fsync(f.fileno())

    def record_timestamp(self, user_id, t, soids):
        for soid in soids:
            user_idx = self.user_ids.index(user_id)
            self.timestamps[soid] = (t, user_idx)
        with open(self.timestamp_path, 'w') as f:
            json.dump(self.timestamps, f)
            f.flush()
            os.fsync(f.fileno())

    def add_edge(self, user, ID1, ID2):
        t = time.time()
        logging.info(f'add_edge({user}, {ID1}, {ID2})')
        edge_created = False
        edge = self.ssv_graph.edge(self.vertex_idx(ID1), self.vertex_idx(ID2))
        if edge is None:
            edge_created = True
            self.ssv_graph.add_edge(self.vertex_idx(ID1), self.vertex_idx(ID2), add_missing=False)
            self.record_action(user, t, '+', (ID1, ID2))
            self.record_timestamp(user, t, (ID1, ID2))
        logging.info(f'add edge took {time.time() - t:.2} s')
        return edge_created

    def del_edge(self, user, ID1, ID2):
        t = time.time()
        logging.info(f'del_edge({user}, {ID1}, {ID2})')
        edge = self.ssv_graph.edge(self.vertex_idx(ID1), self.vertex_idx(ID2))
        edge_found = False
        if edge is not None:
            edge_found = True
            source = self.ssv_graph.vp['seg_ids'][edge.source()]
            target = self.ssv_graph.vp['seg_ids'][edge.target()]
            self.ssv_graph.remove_edge(edge)
            self.record_action(user, t, '-', (ID1, ID2))
            self.record_timestamp(user, t, (ID1, ID2))
        logging.info(f'del_edge took {time.time() - t:.2} s')
        return edge_found

    def add_anchor_note(self, user_id, anchor_id, note):
        t = time.time()
        try:
            self.anchors[anchor_id]['note'] = note
        except KeyError:
            return False
        self.record_action(user_id, t, 'note', (anchor_id, note))
        self.record_timestamp(user_id, t, (anchor_id,))
        return True

    def add_anchor(self, user_id, anchor_id):
        t = time.time()
        if anchor_id in self.anchors:
            return False
        if anchor_id not in self.id_map:
            return False
        self.anchors[anchor_id] = {}
        self.record_action(user_id, t, 'anchor_add', (anchor_id,))
        self.record_timestamp(user_id, t, (anchor_id,))
        logging.info(f'add_anchor took {time.time() - t:.2} s')
        return True

    def delete_anchor(self, user_id, anchor_id):
        t = time.time()
        if anchor_id not in self.anchors or 'nuc' in self.anchors[anchor_id]:
            return False
        del self.anchors[anchor_id]
        self.record_action(user_id, t, 'anchor_del', (anchor_id,))
        self.record_timestamp(user_id, t, (anchor_id,))
        logging.info(f'delete_anchor took {time.time() - t:.2} s')
        return True

    def dfs(self, vertex_idx):
        t = time.time()
        todo = {vertex_idx}
        visited = set()
        while len(todo) > 0:
            if time.time() - t > 1:
                raise DFSTimeout()
            next_vertex = todo.pop()
            visited.add(next_vertex)
            todo |= set(self.ssv_graph.iter_all_neighbors(self.ssv_graph.vertex(next_vertex))) - visited
        logging.info(f'dfs took {time.time() - t:.2} s')
        return visited

    def dfs_wrapper(self, vertex_idx):
        try:
            return self.dfs(vertex_idx)
        except DFSTimeout:
            t = time.time()
            vertex_indices = np.append(search.dfs_iterator(self.ssv_graph, source=vertex_idx, array=True), [vertex_idx])
            cc = np.unique(vertex_indices)
            logging.info(f'graphtool dfs took {time.time() - t:.2} s')
            return cc

    def get_anchors(self, cc):
        return {sv: self.anchors[sv] for sv in self.anchors.keys() & set(cc)}

    def get_timestamp(self, cc):
        svs = self.timestamps.keys() & cc
        timestamp = 0
        user = None
        for sv in svs:
            if timestamp < self.timestamps[sv][0]:
                timestamp = self.timestamps[sv][0]
                user = self.user_names[self.timestamps[sv][1]]
        return timestamp, user

    def get_cc(self, SSVIDs):
        t = time.time()
        logging.info(f'get_cc({SSVIDs})')
        ccs = []
        for SSVID in SSVIDs:
            vertex_idx = self.vertex_idx(SSVID)
            seg_ids = self.ssv_graph.vp['seg_ids']
            ccs.append([seg_ids[v] for v in self.dfs_wrapper(vertex_idx)])
            logging.info(f'get_cc (cc size: {len(ccs)}) took {time.time() - t:.2} s')
        return {'groups': [{'groupMembers': cc} for cc in ccs]}

    def get_mergelist(self, ssvid, whole_cc):
        t = time.time()
        logging.info(f'get_mergelist({ssvid})')
        vertex_idx = self.vertex_idx(ssvid)
        supervoxelIds = {ssvid}
        if not whole_cc:
            supervoxelIds.update({int(neighbor_id) for _, neighbor_id in self.ssv_graph.get_all_neighbors(self.vertex_idx(ssvid), [self.ssv_graph.vp['seg_ids']])})
        else:
            cc = self.dfs_wrapper(vertex_idx)
            cc_mask = self.ssv_graph.new_vp('bool', val=False)
            cc_mask.a = np.zeros(shape=self.ssv_graph.num_vertices())
            cc_mask.a[np.array(list(cc), dtype=int)] = True
            self.ssv_graph.set_vertex_filter(cc_mask)
            for v_idx, seg_id, in self.ssv_graph.iter_vertices(vprops=(self.ssv_graph.vp['seg_ids'],)):
                supervoxelIds.add(seg_id)
            self.ssv_graph.clear_filters()
        mergelist = f'{min(supervoxelIds)} 0 1 {" ".join(str(sv) for sv in supervoxelIds)}\n\n\n\n'
        recommended_lod = get_recommended_lod(self, supervoxelIds)
        timestamp, user = self.get_timestamp(supervoxelIds)
        anchors = self.get_anchors(supervoxelIds)
        logging.info(f'get_mergelist (cc size: {len(supervoxelIds)}) took {time.time() - t:.2} s')
        return {'mergelist': mergelist, 'anchors': {sv: props.get('nuc', 0) for sv, props in  anchors.items()}, 'anchor': anchors, 'timestamp': timestamp, 'user': user, 'recommended_lod': recommended_lod}

    def get_graph(self, SSVID):
        cc = self.dfs_wrapper(self.vertex_idx(SSVID))
        edges = []
        for vertex_idx in cc:
            neighbors = self.ssv_graph.get_all_neighbors(vertex_idx, [self.ssv_graph.vp['seg_ids']])
            edges += [{'first': f'{neighbor_id}', 'second': f'{SSVID}'} for neighbor, neighbor_id in neighbors]
        return {'edge': edges}

    def get_fragments(self, SSVID, whole_cc):
        t = time.time()
        logging.info(f'get_fragments({SSVID}, {whole_cc})')
        vertex_idx = self.vertex_idx(SSVID)
        fragment_keys = []
        supervoxelIds = []
        cc = self.dfs_wrapper(vertex_idx) if whole_cc else [vertex_idx]
        cc_mask = self.ssv_graph.new_vp('bool', val=False)
        cc_mask.a = np.zeros(shape=self.ssv_graph.num_vertices())
        cc_mask.a[np.array(list(cc), dtype=int)] = True
        self.ssv_graph.set_vertex_filter(cc_mask)
        for v_idx, seg_id, frag_keys in self.ssv_graph.iter_vertices(vprops=(self.ssv_graph.vp['seg_ids'], self.ssv_graph.vp['frag_keys'])):
            frag_keys = list(frag_keys)
            supervoxelIds += [seg_id] * len(frag_keys)
            fragment_keys += frag_keys
        self.ssv_graph.clear_filters()
        timestamp, user = self.get_timestamp(supervoxelIds)
        anchors = self.get_anchors(supervoxelIds)
        logging.info(f'get_fragments (cc size: {len(cc)}) took {time.time() - t:.2} s')
        return {'fragmentKey': fragment_keys, 'supervoxelId': supervoxelIds, 'anchors': {sv: props.get('nuc', 0) for sv, props in  anchors.items()}, 'anchor': anchors, 'timestamp': timestamp, 'user': user}

    def get_neighbors(self, SSVIDs):
        t = time.time()
        logging.info(f'get_neighbors ids[0]: {SSVIDs[0]}, #{len(SSVIDs)}')
        edges = []
        for ssvid in SSVIDs:
            neighbors = self.ssv_graph.get_all_neighbors(self.vertex_idx(ssvid), [self.ssv_graph.vp['seg_ids']])
            edges += [{'first': f'{neighbor_id}', 'second': f'{ssvid}'} for neighbor, neighbor_id in neighbors]
        logging.info(f'get_neighbors took {time.time() - t:.2} s')
        return {'edge': edges}
