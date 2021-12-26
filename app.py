#Các thư viện cần thiết
from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import preprocessor as p

#Load transform
transform = pickle.load(open('transform_2.pkl','rb'))

#load model
model = pickle.load(open('model_svm.pkl','rb'))



#Hàm clean text
#set option để xóa URL và emoji
p.set_options(p.OPT.URL, p.OPT.EMOJI)

def clean_text(text):
  #Chuyển các từ thành chữ viết thường
  text = text.lower()

  #Sử dụng tweet-preprocessor để làm sạch tweet
  text = p.clean(text)

  #Loại bỏ stop word
  text = " ".join([w for w in text.split() if w not in STOP_WORDS])

  #Xóa các dấu thừa
  text = re.sub(r'[^\w\s]','',text)

  #Xóa số và các từ có chứa số
  text = re.sub(r'\w*\d\w*','',text)

  #Xóa khoảng trắng dư thừa
  text = re.sub(r'\s+',' ', text)
  return text

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')


@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['mail']
        if not text:
            error_message = 'Please type in your text'
            return render_template('index.html', error=error_message)
        else:
            cleaned_text = clean_text(text)
            message = [cleaned_text]
            test_tfidf = transform.transform(message)
            test_tfidf = test_tfidf.toarray()
            text_predict = model.predict(test_tfidf)
            return render_template('index.html', data=text_predict, value=text)


if __name__ == '__main__':
    app.run(debug=True)
