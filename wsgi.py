#!/usr/bin/python
import os

virtualenv = os.path.join(os.environ['OPENSHIFT_PYTHON_DIR'], 'virtenv', 'bin', 'activate_this.py')
try:
    execfile(virtualenv, dict(__file__=virtualenv))
except IOError:
    print "source activate failed"
    pass
#
# IMPORTANT: Put any additional includes below this line.  If placed above this
# line, it's possible required libraries won't be in your searchable path
#
from keywordcounter import app as application
#
# Below for testing only
#
if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('localhost', 8051, application)
    # Wait for a single request, serve it and quit.
    httpd.serve_forever()
