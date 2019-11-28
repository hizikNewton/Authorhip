from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap
from CF_Model.tests import test_predict
import os

# NLP Packages
from textblob import TextBlob
import random 
import time

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST' and  request.form['rawtext']:
        rawtext = request.form['rawtext']
        result = test_predict.test_prediction(rawtext)
        received_text = rawtext[0:200]+'...'
        accuracy = f"{100-(result['acc']*100):.2f}%"
        predictions = result['predictions']
        dfv = result['dfv']
        end = time.time()
        final_time = f'{end-start:.3f}'
        return render_template('index.html',final_time = final_time,received_text = received_text,predictions = predictions,accuracy = accuracy,dfv= dfv)
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)