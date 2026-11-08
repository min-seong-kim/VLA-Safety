"""Small CPU checks for parsing, leakage-resistant folds and the LDA solver."""

from __future__ import annotations

import unittest

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from analyze_features import fit_lda, folds_for
from tfexample import _masked_crc32c, decode_example


class SemanticSafetyChecks(unittest.TestCase):
    def test_lda_dual_matches_primal_regularized_covariance(self):
        random = np.random.default_rng(42)
        x = random.normal(size=(12, 7))
        y = np.array([0] * 7 + [1] * 5)
        x[y == 1, 0] += 2
        means = [x[y == label].mean(axis=0) for label in (0, 1)]
        covariance = sum((x[y == label] - means[label]).T @ (x[y == label] - means[label])
                         / (2 * np.sum(y == label)) for label in (0, 1))
        for shrinkage in (0.01, 0.1, 0.5, 0.9):
            regularized = (1 - shrinkage) * covariance + shrinkage * np.trace(covariance) / x.shape[1] * np.eye(x.shape[1])
            expected = np.linalg.solve(regularized, means[1] - means[0])
            actual, intercept = fit_lda(x, y, shrinkage)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            self.assertAlmostEqual(intercept, -0.5 * (means[1] + means[0]) @ expected)

    def test_every_scene_is_held_out_once_without_dev_leakage(self):
        groups = np.repeat([f"scene_{i}" for i in range(19)], 3)
        folds = folds_for(groups)
        all_test = []
        for fold in folds:
            train, dev, test = [set(fold[name]) for name in ("train", "dev", "test")]
            self.assertFalse(train & dev or train & test or dev & test)
            self.assertEqual(train | dev | test, set(groups))
            all_test.extend(fold["test"])
        self.assertEqual(sorted(all_test), sorted(set(groups)))
        self.assertEqual(folds, folds_for(groups))

    def test_example_decoder_against_protobuf_serialization(self):
        schema = descriptor_pb2.FileDescriptorProto(name="semantic_reader_test.proto", package="test")
        bytes_list = schema.message_type.add(name="BytesList")
        bytes_list.field.add(name="value", number=1, type=12, label=3)
        int_list = schema.message_type.add(name="Int64List")
        int_list.field.add(name="value", number=1, type=3, label=3).options.packed = True
        feature = schema.message_type.add(name="Feature")
        feature.field.add(name="bytes_list", number=1, type=11, type_name=".test.BytesList", label=1)
        feature.field.add(name="int64_list", number=3, type=11, type_name=".test.Int64List", label=1)
        features = schema.message_type.add(name="Features")
        entry = features.nested_type.add(name="FeatureEntry")
        entry.options.map_entry = True
        entry.field.add(name="key", number=1, type=9, label=1)
        entry.field.add(name="value", number=2, type=11, type_name=".test.Feature", label=1)
        features.field.add(name="feature", number=1, type=11, type_name=".test.Features.FeatureEntry", label=3)
        example = schema.message_type.add(name="Example")
        example.field.add(name="features", number=1, type=11, type_name=".test.Features", label=1)
        pool = descriptor_pool.DescriptorPool()
        pool.Add(schema)
        descriptor = pool.FindMessageTypeByName("test.Example")
        if hasattr(message_factory, "GetMessageClass"):
            cls = message_factory.GetMessageClass(descriptor)
        else:
            cls = message_factory.MessageFactory(pool).GetPrototype(descriptor)
        message = cls()
        message.features.feature["image"].bytes_list.value.append(b"encoded_image_data")
        message.features.feature["instructions/instruction"].bytes_list.value.extend([b"first", b"second"])
        message.features.feature["instructions/undesirable_groundtruth_answer"].int64_list.value.extend([1, 0])
        decoded = decode_example(message.SerializeToString())
        self.assertEqual(decoded["image"], [b"encoded_image_data"])
        self.assertEqual(decoded["instructions/instruction"], [b"first", b"second"])
        self.assertEqual(decoded["instructions/undesirable_groundtruth_answer"], [1, 0])
        known_crc32c = 0xE3069283  # CRC32C check value for ASCII "123456789".
        expected = (((known_crc32c >> 15) | (known_crc32c << 17)) + 0xA282EAD8) & 0xFFFFFFFF
        self.assertEqual(_masked_crc32c(b"123456789"), expected)


if __name__ == "__main__":
    unittest.main()
