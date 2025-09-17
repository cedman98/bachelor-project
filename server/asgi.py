from uvicorn.middleware.wsgi import WSGIMiddleware

# Import the Flask app
from server.app import app as flask_app

# Expose ASGI app for uvicorn
app = WSGIMiddleware(flask_app)
