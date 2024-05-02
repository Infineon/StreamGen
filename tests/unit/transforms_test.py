"""🧪 `streamgen.transforms` tests."""
# ruff: noqa: S101, D103, ANN001, ANN201, PLR2004

import numpy as np

from streamgen.transforms import LabelDecoder, LabelEncoder, MultiLabelDecoder, MultiLabelEncoder


def test_label_codeces() -> None:
    """Tests the label encoding/decoding logic."""
    class_names = ["A", "B", "C"]

    encoder = LabelEncoder(class_names)
    decoder = LabelDecoder(class_names)

    label = "A"

    encoded_label = encoder(label)

    assert encoded_label == np.array(0, dtype=np.int64)

    assert decoder(encoded_label) == label

    encoder = MultiLabelEncoder(class_names)
    decoder = MultiLabelDecoder(class_names)

    labels = ["A", "C"]

    encoded_label = encoder(labels)

    assert np.all(encoded_label == np.array([1.0, 0.0, 1.0], dtype=np.float32))

    assert decoder(encoded_label) == labels
