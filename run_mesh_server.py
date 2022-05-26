__author__ = ['My-Tien Nguyen', 'Norbert Pfeiler', 'Fabian Svara', 'Jörgen Kornfeld']
__copyright__ = "Copyright (C) 2022 ariadne.ai ag"
__license__ = "AGPL v3"
__version__ = "0.1"


import argparse
import logging
import mesh_server
from mesh_server import MeshServer, app, wrapper
from shared import setup_logging

setup_logging('mesh_logs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh server')
    parser.add_argument('--lod_list_path', type=str, default='lod.txt')
    parser.add_argument('--mesh_path', type=str, default=f'file://meshes')
    parser.add_argument('--user_db_path', type=str, default='users.txt')
    parser.add_argument('--port', type=int,
                        help='Port to connect to the mesh server.',  default=12000)
    parser.add_argument('--host', type=str, help='IP to bind to.', default='0.0.0.0')

    args = parser.parse_args()

    mesh_server.mesh_server = MeshServer(user_db_path=args.user_db_path, mesh_path=args.mesh_path, lod_list_path=args.lod_list_path)
    wrapper.server = mesh_server.mesh_server

    logging.info(f'Mesh server running at {args.host}:{args.port}.')
    app.run(host=args.host,
            port=args.port,
            threaded=True)

else:
    mesh_server.mesh_server = MeshServer()
    wrapper.server = mesh_server.mesh_server
