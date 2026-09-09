from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOISE_CLASS_NAMES = {0: 'stationary', 1: 'non_stationary', 2: 'impulsive'}


def build_logistic_regression(C=1.0, max_iter=1000, random_state=42):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=max_iter,
            solver='lbfgs',
            random_state=random_state,
        )),
    ])


def build_random_forest(n_estimators=200, max_depth=None, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state,
    )


def build_hist_gradient_boosting(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    l2_regularization=0.1,
    random_state=42
):
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        l2_regularization=l2_regularization,
        class_weight='balanced',
        random_state=random_state,
        verbose=0,
    )


def get_all_baselines():
    return {
        'logistic_regression': build_logistic_regression(),
        'random_forest': build_random_forest(),
        'hist_gradient_boosting': build_hist_gradient_boosting(),
    }
