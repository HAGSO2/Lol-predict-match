"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import Prediction_view.views
import Prediction_view.load_data