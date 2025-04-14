#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

# Fit specific
from scipy.optimize import curve_fit
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures as PF
from lmfit import create_params, minimize

# Error handling
import traceback
import warnings
warnings.filterwarnings("ignore")

def gaus(x, *p0):
    N, mu, sigma = p0
    return N*np.exp(-(x-mu)**2/(2.0*sigma**2))

def linear(x, *p0):
    a, b = p0
    return a + b*x

def quad(x, *p0):
    a, b, c = p0
    return a + b*x + c*x**2

def cubic(x, *p0):
    a, b, c, d = p0
    return a + b*x + c*x**2 + d*x**3

def poly(deg, x):
    p = PF(deg)
    return PF.fit_transform(x)

spy_models = {
    'linear':(linear, [12, -0.15]),
    'quad':(quad, [12, -0.15, -0.1]),
    'cubic':(cubic, [12, -0.15, -0.1, -0.1 ]),
}
sci_models = {
    'linearSVR': [LinearSVR, {'epsilon':1e-1, 'loss':'squared_epsilon_insensitive'}, 2, [12, -0.15]],
    'linearRidge':[Ridge, {'alpha':1e-1}, 2, None, [12, -0.15]],
    'linearLasso':[Ridge, {'alpha':1e-1}, 2, None, [12, -0.15]],
    'quadRidge':[Ridge, {'alpha':1e-1}, 3, poly, [12, -0.15, -0.1]],
    'quadLasso':[Ridge, {'alpha':1e-1}, 3, poly, [12, -0.15, -0.1]],
    'cubicRidge':[Ridge, {'alpha':1e-1}, 3, poly, [12, -0.15, -0.1, -0.1]],
    'cubicLasso':[Ridge, {'alpha':1e-1}, 3, poly, [12, -0.15, -0.1, -0.1]],
}
lm_models = {
    'linearLM': [linear, {'a':{'value':12}, 'b':{'value':-0.15, 'max':0}}],
    'quadLM': [quad, {'a':{'value':12}, 'b':{'value':-0.15}, 'c':{'value':-0.1, 'max':0},}],
    'cubicLM': [cubic, {'a':{'value':12}, 'b':{'value':-0.15, 'max':0}, 'c':{'value':-0.1, 'max':0}, 'd':{'value':-0.1, 'max':0}}],
    #'cubicLM': [cubic, {'a':{'value':0}, 'b':{'value':0, 'max':0}, 'c':{'value':0.0018, 'max':0.01}, 'd':{'value':-0.534, 'max':0}}],
}

def train(model_name, x, y, yerr = None, verbose = False):
    fail = [-99999]*4
    if model_name in spy_models:
        try:
            model, p0 = spy_models[model_name]
            tw_corr, var_matrix = curve_fit(
                model,
                x,
                y,
                #check_finite=False,
                sigma=yerr,
                p0=p0,)
            tw_corr = tw_corr.tolist()
        except ValueError:
            print("Fit fails because of only empty bins")
            if verbose:
                traceback.print_exc()
            tw_corr = fail
        except TypeError:
            print("Fit fails because of too few filled bins")
            if verbose:
                traceback.print_exc()
            tw_corr = fail
        except KeyError:
            print(f"Probably no events for ")
            if verbose:
                traceback.print_exc()
            tw_corr = fail
    elif model_name in sci_models:
        try:
            model_class, params, limit, process, p0 = sci_models[model_name]
            model = model_class(**params)
            if process:
                x = process(limit - 1, x.reshape(-1, 1))
                model.fit(x, y)
            else:
                model.fit(x.reshape(-1, 1), y)
            tw_corr = model.intercept_.tolist() + model.coef_.tolist()
        except:
            if verbose:
                traceback.print_exc()
            print(f"TOT {len(x)}, DT {len(y)}")
            print('Something Failed, probably insufficient data. Returning ridiculous values')
            tw_corr = fail
    elif model_name in lm_models:
        model, p0 = lm_models[model_name]
        params = create_params(**p0)
        p0 = [p0[p]['value'] for p in p0]
        try:
            def residual(pars, x, data = [None]):
                p = [pars[p] for p in pars]
                if all(data):
                    return model(x, *p) - data
                else:
                    return model(x, *p)
            out = minimize(residual, params, args=(x,), kws={'data':y})
            tw_corr = [out.params[a] for a in out.params]
        except:
            if verbose:
                traceback.print_exc()
            print(f"TOT {len(x)}, DT {len(y)}")
            print('Something Failed, probably insufficient data. Returning ridiculous values')
            tw_corr = fail
    else:
        print('Model not known. Returning linear fit default values')
        tw_corr = spy_models['linear'][1]
    if len(tw_corr) != 4:
        return tw_corr + [0]*(4 - len(tw_corr))
    return tw_corr 

def predict(model_name, x, model_params):
    if model_name in spy_models:
        model, p0 = spy_models[model_name]
        if len(model_params) > len(p0):
            temp = [model_params[i] for i in range(len(p0))]
            return model(x, *temp)
        else:
            return model(x, *model_params)
    elif model_name in sci_models:
        if len(x) < 1:
            return []
        model_class, params, limit, process, p0 = sci_models[model_name]
        model = model_class(**params)
        if len(model_params) > len(p0):
            temp = [model_params[i] for i in range(len(p0))]
            model.coef_ = np.array(temp)[1:]
            model.intercept_ = np.array(temp)[0]
        else:
            model.coef_ = np.array(model_params)[1:]
            model.intercept_ = np.array(model_params)[0]
        if process:
            x = process(limit - 1, x)
            return model.predict(x)
        else:
            return model.predict(x.reshape(-1, 1))
    elif model_name in lm_models:
        model, p0 = lm_models[model_name]
        if len(model_params) > len(p0):
            temp = [model_params[i] for i in range(len(p0))]
            return model(x, *temp)
        return model(x, *model_params)

def calc_timewalk_corrections_unbinned(raw_data, row, col, model):
    #print(f"Timewalk {row=}, {col=}")
    x = np.array(raw_data[row][col]['tot'])
    y = np.array(raw_data[row][col]['dt'])
 
    if len(x > 50):
        xn = 2
        '''
        h, e, _ = plt.hist(x, bins = 20)
        i = np.argmax(h)
        c = np.mean([e[i], e[i+1]])
        '''
        c = np.median(x)
        s = np.std(x)
        xidx = (x < c + xn*s)&(x > c - xn*s)
    elif len(x) == 0:
        return [-99999]*4
    else:
        xidx = np.array([True]*len(x))

    if len(y) > 50:
        yn = 16
        '''
        h, e, _ = plt.hist(y, bins = 20)
        i = np.argmax(h)
        c = np.mean([e[i], e[i+1]])
        '''
        c = np.median(y)
        s = np.std(y)
        yidx = (y < c + yn*s)&(y > c - yn*s)
    else:
        yidx = np.array([True]*len(x))
    x = x[xidx&yidx]
    y = y[xidx&yidx]
    return train(model, x, y)


def calc_timewalk_corrections_unbinned_general(x, y, model):
    #print(f"Timewalk {row=}, {col=}")
    if len(x > 50):
        xn = 2
        '''
        h, e, _ = plt.hist(x, bins = 20)
        i = np.argmax(h)
        c = np.mean([e[i], e[i+1]])
        '''
        c = np.median(x)
        s = np.std(x)
        xidx = (x < c + xn*s)&(x > c - xn*s)
    elif len(x) == 0:
        return [-99999]*4
    else:
        xidx = np.array([True]*len(x))

    if len(y) > 50:
        yn = 16
        '''
        h, e, _ = plt.hist(y, bins = 20)
        i = np.argmax(h)
        c = np.mean([e[i], e[i+1]])
        '''
        c = np.median(y)
        s = np.std(y)
        yidx = (y < c + yn*s)&(y > c - yn*s)
    else:
        yidx = np.array([True]*len(x))
    x = x[xidx&yidx]
    y = y[xidx&yidx]
    return train(model, x, y)

def calc_timewalk_corrections(dt_tot_hist, row, col, model):
    #print(f"Timewalk {row=}, {col=}")
    timewalk_hist = dt_tot_hist[{'row':row, 'col':col}].profile('dt')
    x = timewalk_hist.axes.centers[0]
    y = timewalk_hist.values()
    idx = y > -5000
    x = x[idx]
    y = y[idx]
    return train(model, x, y, yerr = np.ones_like(y)*0.1)

def calc_timewalk_corrections_2dbinned(dt_tot_hist, row, col, model):
    print(f"Timewalk {row=}, {col=}")
    timewalk_hist = dt_tot_hist[{'row':row, 'col':col}]
    ye = timewalk_hist.axes.centers[0].reshape(1, -1)[0]
    xe = timewalk_hist.axes.centers[1][0]
    ee = timewalk_hist.counts()
    x = []
    y = []
    e = []
    for i in range(len(xe)):
        for j in range(len(ye)):
            if ee[j, i] > 0:
                x.append(xe[i])#np.mean([xe[i], xe[i+1]]))
                y.append(ye[j])#np.mean([ye[j], ye[j+1]]))
                e.append(1/np.sqrt(ee[j, i]))
    return train(model, np.array(x), np.array(y))#, yerr = np.array(e))

