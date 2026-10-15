import numpy as np

from src.models import custom_cnn, resnet34


def test_custom_cnn_output_shape():
    model = custom_cnn.build(input_shape=(28, 28, 1), n_classes=10)
    out = model(np.random.rand(2, 28, 28, 1).astype("float32"))
    assert out.shape == (2, 10)


def test_custom_cnn_probabilities_sum_to_one():
    model = custom_cnn.build(input_shape=(28, 28, 1), n_classes=10)
    out = model(np.random.rand(1, 28, 28, 1).astype("float32")).numpy()
    np.testing.assert_allclose(out.sum(axis=1), [1.0], atol=1e-5)


def test_resnet34_output_shape():
    # Use a small input so the test runs fast
    model = resnet34.build(input_shape=(56, 56, 1), n_classes=10)
    out = model(np.random.rand(2, 56, 56, 1).astype("float32"))
    assert out.shape == (2, 10)


def test_resnet34_probabilities_sum_to_one():
    model = resnet34.build(input_shape=(56, 56, 1), n_classes=5)
    out = model(np.random.rand(1, 56, 56, 1).astype("float32")).numpy()
    np.testing.assert_allclose(out.sum(axis=1), [1.0], atol=1e-5)
