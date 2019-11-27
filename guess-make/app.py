# -- coding: utf-8 --
import os
import pandas as pd
import random

from flask import Flask, render_template, flash, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from CarModelClassifier.estimation import prediction


STATIC_FOLDER = 'static'
DATA_FOLDER = os.path.join(STATIC_FOLDER, 'images')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'very hard to guess string'
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['UPLOAD_FOLDER'] = DATA_FOLDER
out_df = prediction()
bootstrap = Bootstrap(app)

choices = [('Audi A3 Cabriolet', 'Audi A3 Cabriolet'), ('Audi Q5', 'Audi Q5'),
           ('BMW Series 1 Coupe', 'BMW Series 1 Coupe'), ('BMW Z4 Roadster', 'BMW Z4 Roadster'),
           ('Fiat Doblo', 'Fiat Doblo'), ('Fiat Panda', 'Fiat Panda'), ('Fiat 500L', 'Fiat 500L'),
           ('Ford Explorer', 'Ford Explorer'), ('Ford Fiesta', 'Ford Fiesta'), ('Jaguar E Pace', 'Jaguar E Pace'),
           ('Jaguar F Type', 'Jaguar F Type'), ('Jaguar XF Sportbrake', 'Jaguar XF Sportbrake'),
           ('Jeep Gladiator', 'Jeep Gladiator'), ('Jeep Grand Cherokee', 'Jeep Grand Cherokee'),
           ('Land Range Rover Evoque', 'Land Range Rover Evoque'), ('Mada CX5', 'Mazda CX5'),
           ('Mazda Model 3', 'Mada Model 3'), ('Mazda MX5', 'Mazda MX5'), ('Mini Cabrio', 'Mini Cabrio'),
           ('Mini Cooper Clubman', 'Mini Cooper Clubman'), ('Mitsubishi ASX', 'Mitsubishi ASX'),
           ('Mitsubishi Outlander', 'Mitsubishi Outlander'), ('Mitsubishi Pajero', 'Mitsubishi Pajero'),
           ('Nissan Kia', 'Nissan Kia'), ('Nissan Micra', 'Nissan Micra'), ('Nissan Qashqai', 'Nissan Qashqai'),
           ('Opel Antara', 'Opel Antara'), ('Opel Meriva', 'Opel Meriva'), ('Opel Mokka', 'Opel Mokka'),
           ('Opel Zafira', 'Opel Zafira'), ('Porsche Boxster Spyder', 'Porsche Boxster Spyder'),
           ('Porsche Cayenne Turbo', 'Porsche Cayenne Turbo'), ('Porsche Cayman', 'Porsche Cayman'),
           ('Porsche Taycan', 'Porsche Taycan'), ('Smart Fortwo', 'Smart Fortwo'), ('Toyota Aygo', 'Toyota Aygo'),
           ('Toyota Corolla', 'Toyota Corolla'), ('Toyota Yaris', 'Toyota Yaris'),
           ('Volswagen Golf GTI', 'Volswagen Golf GTI'),  ('Volswagen Jetta', 'Volswagen Jetta'),
           ('Volswagen Up', 'Volswagen Up')]


class PastebinEntry(FlaskForm):
    make = SelectField(u'Guess the Model',
                       choices=choices,
                       validators=[
                           DataRequired(u'DataRequired')]
                       )
    submit = SubmitField(u'Submit')


def get_path(r):
    image_name = f'{str(r).zfill(6)}.jpg'
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    return img_path, image_name


def get_make(r):
    img = f'{str(r).zfill(6)}.jpg'
    # return make and model from image name
    label_df = pd.read_csv("../custom_evaluation/test_labels.csv")
    info_df = pd.read_csv("../data/labels/models_info_new.csv")
    total_df = label_df.merge(info_df, on=['model_label'])
    brand = total_df.loc[total_df['fname'] == img, :].reset_index()['brand'][0]
    model = total_df.loc[total_df['fname'] == img, :].reset_index()['model'][0]
    res = str(brand) + " " + str(model)
    return res


@app.route('/')
def index():
    label_df = pd.read_csv("../custom_evaluation/test_labels.csv")
    r = random.randint(1, len(label_df))
    img_path, image_name = get_path(r)
    session['img_path'] = img_path
    session['image_name'] = image_name
    make = get_make(r)
    assert type(make) == str
    session['make'] = make
    return render_template('index.html', img_path=img_path)


@app.route('/guess', methods=['GET', 'POST'])
def guess():
    img_path = session['img_path']
    image_name = session['image_name']
    result = str(session['make'])
    form = PastebinEntry()
    brand = out_df.loc[out_df.filename == image_name, :].reset_index()['brand'][0]
    model = out_df.loc[out_df.filename == image_name, :].reset_index()['model'][0]
    predicted_car = f'{brand} {model}'
    if form.validate_on_submit():
        answer = str(form.make.data)
        if answer != result:
            flash(f'Failed! You predicted {answer}, while the actual model is {result}.',
                  'danger')
            flash(f'\nOur neural network predicted {predicted_car}', 'info')
            return redirect(url_for('index', img_path=img_path))
        else:
            flash(f'Success!', 'success')
            flash(f'\nOur neural network predicted {predicted_car}', 'info')
            return redirect(url_for('index', img_path=img_path))
    return render_template('guess.html', form=form, img_path=img_path)


if __name__ == '__main__':
    app.run()