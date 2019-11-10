# -*- coding: utf-8 -*-
import os
import pandas as pd
import random

from flask import Flask, render_template, flash, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap


STATIC_FOLDER = 'static'
PIC_FOLDER = os.path.join(STATIC_FOLDER, 'images')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'very hard to guess string'
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['UPLOAD_FOLDER'] = PIC_FOLDER

bootstrap = Bootstrap(app)


class PastebinEntry(FlaskForm):
    make = SelectField(u'Guess the Make', choices=[('Mini Cooper', 'Mini Cooper'),
                                                   ('Jaguar', 'Jaguar'), ('Porsche', 'Porsche')],
                       validators=[
                           DataRequired(u'DataRequired')]
                       )
    submit = SubmitField(u'Submit')


def get_path(r):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{str(r).zfill(5)}.jpg')
    return img_path


def get_make(r):
    img = f'{str(r).zfill(5)}.jpg'
    labels_path = str(os.path.join(STATIC_FOLDER, 'images/')) + 'labels_prova.csv'
    labels = pd.read_csv(labels_path)
    make = labels[labels.name == img]['make'][labels[labels.name == img]['make'].index[0]]
    return str(make)


@app.route('/')
def index():
    session['times'] = 1
    r = random.randint(8274, 8284)  # TODO change this to randint(0, len(labels))
    img_path = get_path(r)
    session['img_path'] = img_path
    make = get_make(r)
    assert type(make) == str
    session['make'] = make
    return render_template('index.html', img_path=img_path)


@app.route('/guess', methods=['GET', 'POST'])
def guess():
    # times = session['times']
    img_path = session['img_path']
    result = str(session['make'])
    # print(result)
    form = PastebinEntry()
    if form.validate_on_submit():
        # times -= 1
        # session['times'] = times
        # if times < 0:
        #     flash(u'Sorry, you are out of trials', 'danger')
        #     return redirect(url_for('index', img_path=img_path))
        answer = str(form.make.data)
        if answer != result:
            flash(f'Failed! You predicted {answer}, while the actual make is {result}', 'danger')
            # TODO eventually add a line showing what our best net has predicted for this car
            return redirect(url_for('index', img_path=img_path))
        else:
            flash(u'Success!', 'success')
            return redirect(url_for('index', img_path=img_path))
        # return redirect(url_for('guess', img_path=img_path))
    return render_template('guess.html', form=form, img_path=img_path)


if __name__ == '__main__':
    app.run()
