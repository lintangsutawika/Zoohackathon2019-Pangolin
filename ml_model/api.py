import flask
from flask import Flask
import news_scrap
import time

app = Flask(__name__)

@app.route("/scrap_web", methods=["POST"])
def detect():
    ts = time.time()
    ts_str = "{:0<20}".format(str(ts).replace(".",""))
    print("API JALAN !!!")
    keyword = flask.request.form.get("keyword")
    print(keyword)
    out = news_scrap.scrap_news(keyword)
    return(flask.jsonify(out), 200)
