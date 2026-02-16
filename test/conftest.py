"""
Pytest configuration and fixtures.
"""
import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require model loading"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        # Run all tests including integration
        return
    
    skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def model_fixtures():
    """
    Session-scoped fixture for loaded model.
    Only loads model once per test session when integration tests run.
    """
    from app.core.model import load_model, get_tokenizer, get_model
    
    load_model()
    
    return {
        'tokenizer': get_tokenizer(),
        'model': get_model()
    }
