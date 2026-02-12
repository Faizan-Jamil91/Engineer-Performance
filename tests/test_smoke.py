import importlib


def test_import_utils_modules():
    modules = [
        'utils.data_processor',
        'utils.analytics_engine',
        'utils.visualizations',
        'utils.ai_insights',
        'utils.ml_analytics',
        'config.settings',
    ]

    for mod in modules:
        imported = importlib.import_module(mod)
        assert imported is not None
