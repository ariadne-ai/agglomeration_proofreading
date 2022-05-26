__author__ = ['My-Tien Nguyen', 'Norbert Pfeiler', 'Fabian Svara', 'Jörgen Kornfeld']
__copyright__ = "Copyright (C) 2022 ariadne.ai ag"
__license__ = "AGPL v3"
__version__ = "0.1"


import argparse
import os
import logging

from shared import setup_logging
import agglomeration_server
from agglomeration_server import AgglomerationServer, app, wrapper

setup_logging('logs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proofreading server')
    parser.add_argument('--graph_path', type=str,
                        help='Path to the supervoxel graph edge list.', default='test_graph.gt')
    parser.add_argument('--lod_list_path', type=str, default='lod.txt')
    parser.add_argument('--action_db_path', type=str,
                        default='action_db.tsv',
                        help='Path to the action log tsv file.')
    parser.add_argument('--anchor_path', type=str, default='anchors.txt')
    parser.add_argument('--timestamp_path', type=str, default='timestamps.json')
    parser.add_argument('--user_db_path', type=str, default='users.txt')
    parser.add_argument('--dataset_path', type=str, default='confs')
    parser.add_argument('--port', type=int,
                        help='Port to connect to the mesh server.', default=11000)
    parser.add_argument('--host', type=str, help='IP to bind to.', default='0.0.0.0')

    args = parser.parse_args()

    agglomeration_server.server_state = AgglomerationServer(action_db_path=args.action_db_path, anchor_path=args.anchor_path, timestamp_path=args.timestamp_path, user_db_path=args.user_db_path, ssv_edge_list_path=args.graph_path, lod_list_path=args.lod_list_path, dataset_path=args.dataset_path)
    wrapper.server = agglomeration_server.server_state

    logging.info(f'Proofreading server running at {args.host}:{args.port}.')
    app.run(host=args.host,
            port=args.port,
            threaded=True)

else:
    agglomeration_server.server_state = AgglomerationServer(ssv_edge_list_path=os.environ['graph_path'])
    wrapper.server = agglomeration_server.server_state
