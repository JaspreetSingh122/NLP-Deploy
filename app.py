from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle

# Updating on github 10/11/2023git 
filename = 'nlp_model.pkl'

clf = pickle.load(open(filename, 'rb'))

cv=pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

thisdict = {
    0 : "Intern at Advocate",
  1: "Choose Arts",
  2: "Choose Automation Testing",
  3: "Choose Blockchain",
4: "Choose Business Analyst",
  5: "Choose Civil Engineer",
  6: "Choose Data Science",
    7: "Choose Database Field",
  8: "Choose DevOps Engineer",
  9: "Choose DotNet Developer",
    10: "Choose ETL Developer",
    11: "Choose Electrical Engineering",
  12: "Choose HR",
  13: "Choose Hadoop",
    14: "Choose Health and fitness",
  15: "Choose Java Developer",
  16: "Choose Mechanical Engineer",
    17: "Choose Network Security Engineer",
  18: "Choose Operations Manager",
  19: "Choose PMO",
    20: "Choose Python Developer",
  21: "Choose SAP Developer",
  22: "Choose Sales",
    23: "Choose Testing",
  24: "Choose Web Designing"
}

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data=[message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        new_array = my_prediction.astype(np.int)
        a=new_array[0]
    return render_template('result.html',prediction =thisdict[a])

if __name__ == '__main__':
	app.run(debug=True)