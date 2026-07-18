import http.server
import ssl
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# serve the project root (one level up from this server/ folder)
os.chdir(os.path.dirname(SCRIPT_DIR))

server_address = ('', 8888)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(os.path.join(SCRIPT_DIR, 'cert.pem'),
                    os.path.join(SCRIPT_DIR, 'key.pem'))
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

print("HTTPS server running on https://0.0.0.0:8888")
httpd.serve_forever()
