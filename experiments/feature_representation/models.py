"""Inexpensive, fold-fitted representation diagnostics; no quantum changes."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from challenge.classical import reference

VIEWS = ['pca4','pca6','pca8','pca13','raw13','late5_8','selected4of8']


def late_components(X):
    return X[:,4:8]


def representation(name):
    if name not in VIEWS:
        raise ValueError(name)
    steps=[('input_scale',StandardScaler())]
    if name!='raw13':
        n={'pca4':4,'pca6':6,'pca8':8,'pca13':13,'late5_8':8,'selected4of8':8}[name]
        steps += [('pca',PCA(n_components=n,svd_solver='full')),('pc_scale',StandardScaler())]
        if name=='late5_8':steps.append(('select',FunctionTransformer(late_components)))
        if name=='selected4of8':steps.append(('select',SelectKBest(f_classif,k=4)))
    return Pipeline(steps)


def model(name, learner):
    steps=list(representation(name).steps)
    if learner=='logistic':
        clf=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,tol=1e-6)
    elif learner in ('svm_fixed','svm_nested'):
        # Reuse the actual organizer SVC with its fixed C/gamma; preprocessing
        # is supplied by the chosen fold-fitted representation above.
        clf=reference().named_steps['svc']
    else:
        raise ValueError(learner)
    return Pipeline([*steps,('classifier',clf)])


def selected_components(fitted,name):
    if name=='raw13':return None
    if name=='late5_8':return [5,6,7,8]
    if name=='selected4of8':return (fitted.named_steps['select'].get_support(indices=True)+1).tolist()
    return list(range(1,int(name[3:])+1))
