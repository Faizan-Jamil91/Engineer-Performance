"""Lightweight import checker that stubs heavy dependencies.

Run with the project Python interpreter (no pytest required):

    python tests/run_light_imports.py

This script inserts minimal stub modules into sys.modules for packages
like pandas/streamlit/plotly/sklearn and then attempts to import the
project's core Python modules to ensure they load without syntax/import
errors when heavy binary dependencies are absent.
"""
import sys
import types
import importlib
import os
from pathlib import Path


def make_stub(name):
    m = types.ModuleType(name)
    return m


def ensure_modules(names):
    for nm in names:
        if nm in sys.modules:
            continue
        sys.modules[nm] = make_stub(nm)


def main():
    # Ensure project root is on sys.path so top-level packages like `utils` and `config` import correctly
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    heavy = [
        'pandas', 'numpy', 'plotly', 'plotly.express', 'plotly.graph_objects', 'plotly.subplots',
        'streamlit', 'sklearn', 'sklearn.preprocessing', 'sklearn.cluster', 'sklearn.ensemble',
        'sklearn.linear_model', 'google', 'google.generativeai', 'chardet', 'openpyxl', 'xlsxwriter'
    ]

    # Create stub modules and basic submodules for dotted names
    for name in heavy:
        parts = name.split('.')
        for i in range(1, len(parts)+1):
            modname = '.'.join(parts[:i])
            if modname not in sys.modules:
                sys.modules[modname] = make_stub(modname)

    # Provide minimal attributes used at import-time
    # e.g., pandas.DataFrame, numpy.array
    sys.modules['pandas'].DataFrame = object
    sys.modules['pandas'].Timestamp = object
    sys.modules['pandas'].to_datetime = lambda *a, **k: None
    sys.modules['pandas'].Series = object
    sys.modules['numpy'].array = lambda *a, **k: None

    # sklearn stubs
    sklearn_pre = sys.modules.get('sklearn.preprocessing')
    if sklearn_pre is not None:
        class StandardScaler:
            def fit_transform(self, X):
                return X
        sklearn_pre.StandardScaler = StandardScaler

    sklearn_cluster = sys.modules.get('sklearn.cluster')
    if sklearn_cluster is not None:
        class KMeans:
            def __init__(self, *a, **k):
                pass
            def fit_predict(self, X):
                return [0]*len(getattr(X, 'tolist', lambda: X)())
        sklearn_cluster.KMeans = KMeans

    sklearn_ens = sys.modules.get('sklearn.ensemble')
    if sklearn_ens is not None:
        class IsolationForest:
            def __init__(self, *a, **k):
                pass
            def fit_predict(self, X):
                return [1]*len(getattr(X, 'tolist', lambda: X)())
        sklearn_ens.IsolationForest = IsolationForest

    sklearn_lm = sys.modules.get('sklearn.linear_model')
    if sklearn_lm is not None:
        class LinearRegression:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return [0]*len(getattr(X, 'tolist', lambda: X)())
        sklearn_lm.LinearRegression = LinearRegression

    # plotly.subplots stub
    plotly_sub = sys.modules.get('plotly.subplots')
    if plotly_sub is not None:
        def make_subplots(*a, **k):
            class DummyFig:
                def update_layout(self, *a, **k):
                    return self
            return DummyFig()
        plotly_sub.make_subplots = make_subplots

    # plotly.graph_objects minimal stub
    pg = sys.modules.get('plotly.graph_objects')
    if pg is not None:
        class Figure:
            def update_layout(self, *a, **k):
                return self
        pg.Figure = Figure

    # plotly.express colors stub
    px = sys.modules.get('plotly.express')
    if px is not None:
        colors_ns = types.SimpleNamespace()
        colors_ns.qualitative = types.SimpleNamespace(Set3=list())
        colors_ns.sequential = types.SimpleNamespace(Blues=list())
        colors_ns.diverging = types.SimpleNamespace(RdBu=list())
        px.colors = colors_ns

    # dotenv stub
    if 'dotenv' not in sys.modules:
        dotenv = make_stub('dotenv')
        dotenv.load_dotenv = lambda *a, **k: None
        sys.modules['dotenv'] = dotenv

    # streamlit minimal stub functions used during import
    st = sys.modules.get('streamlit')
    if st is not None:
        st.warning = lambda *a, **k: None
        st.markdown = lambda *a, **k: None
        st.info = lambda *a, **k: None

    modules_to_check = [
        'config.settings',
        'utils.data_processor',
        'utils.analytics_engine',
        'utils.visualizations',
        'utils.ai_insights',
        'utils.ml_analytics'
    ]

    failed = []
    for mod in modules_to_check:
        try:
            importlib.import_module(mod)
            print(f"OK: imported {mod}")
        except Exception as e:
            print(f"FAILED: {mod} -> {e}")
            failed.append((mod, str(e)))

    if failed:
        print('\nSome imports failed. See above.')
        sys.exit(2)
    else:
        print('\nLightweight import check passed.')
        sys.exit(0)


if __name__ == '__main__':
    main()
