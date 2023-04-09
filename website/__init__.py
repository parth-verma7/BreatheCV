from flask import Flask
from os import path

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'sfdfb9OfbgLWdsfdbf83j4K4iuosfdbfpO'
    
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    # cre_db(app)
    return app