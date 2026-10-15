from flask import Flask, render_template, request
import model as ml
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    mse = None
    pred = None
    if request.method == 'POST':
        area = float(request.form['area'])
        bedroom = int(request.form['bedroom'])
        bathroom = int(request.form['bathroom'])
        furnish = int(request.form['furnish'])
        algo = int(request.form['algo'])

        data = np.array([area, bedroom, bathroom, furnish]).reshape(1, -1)

        if algo == 1:
            mse, pred = ml.linear_regression(ml.X_train, ml.X_test, ml.y_train, ml.y_test, data)
        elif algo == 2:
            mse, pred = ml.decision_tree(ml.X_train, ml.X_test, ml.y_train, ml.y_test, data)
        elif algo == 3:
            mse, pred = ml.random_forest(ml.X_train, ml.X_test, ml.y_train, ml.y_test, data)
        else:
            mse, pred = ml.support_vector_regressor(ml.X_train, ml.X_test, ml.y_train, ml.y_test, data)

    return render_template('home.html', mse=mse, pred=pred)

if __name__ == '__main__':
    app.run(debug=True)