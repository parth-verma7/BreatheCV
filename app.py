from website import create_app
from flask import Flask, render_template
# from .camera import Video

app=create_app()

if __name__ == '__main__':
    app.run(debug=False)