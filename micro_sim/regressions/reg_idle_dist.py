from micro_sim.main1 import *
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import pandas as pd

Y = {"idle_dist": glob_idle_dist_avg}
X = {
    "demand_density": glob_demand_count_norm,
    "vehicle_density": glob_veh_density,
    "request_rate": glob_request_rate_per_veh_per_min
}

Ydf = pd.DataFrame(Y)
Xdf = pd.DataFrame(X)



def regression1():
    x_train, x_test, y_train, y_test = train_test_split(Xdf, Ydf, test_size=0.2, random_state=42)

    model = LinearRegression().fit(x_train, y_train)

    y_prediction = model.predict(x_test)

    # predicting the accuracy score
    score = r2_score(y_test, y_prediction)
    print()
    print('r2 score is ', score)
    print('mean_sqrd_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared error of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))


def regression2():
    model2 = LinearRegression().fit(Xdf, Ydf)


    r_sq = model2.score(Xdf, Ydf)
    print('coefficient of determination: ', r_sq)
    print('intercept: ', model2.intercept_)
    print('coefficients: ', model2.coef_)


def poly_regression1():
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xdf)
    x_train, x_test, y_train, y_test = train_test_split(x_, Ydf, test_size=0.2, random_state=42)
    model = LinearRegression().fit(x_train, y_train)
    y_prediction = model.predict(x_test)

    # predicting the accuracy score
    score = r2_score(y_test, y_prediction)
    print()
    print('r2 score is ', score)
    print('mean_sqrd_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared error of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))



def poly_regression2():
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(Xdf)
    model = LinearRegression().fit(x_, Ydf)

    r_sq = model.score(x_, Ydf)
    print('coefficient of determination: ', r_sq)
    #print('intercept: ', model.intercept_)
    #print('coefficients: ', model.coef_)

    filename = '/Users/maryia/PycharmProjects/SimulationTrial/estimation models/idle_dist_model'
    pickle.dump(model, open(filename, 'wb'))



#poly_regression1()
poly_regression2()
