from sklearn.dummy import DummyRegressor

from fxensemble import ModelRegistry


def test_custom_model_can_be_registered_and_configured():
    registry = ModelRegistry(seed=11)
    registry.register("constant", lambda parameters: DummyRegressor(**parameters))
    model = registry.create("constant", {"strategy": "constant", "constant": 2.5})
    assert isinstance(model, DummyRegressor)
    assert model.constant == 2.5


def test_registry_rejects_accidental_replacement():
    registry = ModelRegistry()
    try:
        registry.register("ridge", lambda parameters: DummyRegressor())
    except ValueError as error:
        assert "already registered" in str(error)
    else:
        raise AssertionError("Duplicate registration was accepted")
