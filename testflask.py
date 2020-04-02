#!/usr/bin/python3
from flask import Flask, render_template, request
from testclassification import *

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login",methods=['POST'])
def login():
    #receive username and password
    username = request.form.get("username")
    password = request.form.get("pwd")
#we could connect database here to check your username and password

#    rret = functionbbb();
#    print(rret)
    if username == "min" and password == "123":
        return "ok"
    else:
        return render_template("login.html",msg="login failed")

if __name__ == '__main__':
    functionbbb();
    app.run(host='0.0.0.0',debug=True,port=5000)
