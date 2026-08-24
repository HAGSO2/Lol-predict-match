"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from Prediction_view import app

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/prediction_net')
def prediction_net():
    """Renders the prediction page."""
    return render_template(
        'prediction_net.html',
        title='Predict',
        year=datetime.now().year,
        message='View the prediction based on the model'
    )

@app.route('/prediction_boost')
def prediction_boost():
    """Renders the prediction page."""
    return render_template(
        'prediction_boost.html',
        title='Predict',
        year=datetime.now().year,
        message='View the prediction based on the model'
    )