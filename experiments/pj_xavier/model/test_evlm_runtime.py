"""CPU-only contract tests. The optional MLX test selects mx.cpu before arrays."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import evlm_runtime as ev


def eeg(value=0.0):
    return {"schema": ev.EEG_SCHEMA, "units": "V", "sample_rate_hz": 256,
            "channel_order": list(ev.CHANNELS), "values": [[value] * 448 for _ in range(8)],
            "synthetic": True}


def raw_descriptor():
    return {"task_kind": "quantized_eeg_reconstruction", "quantization_schema": ev.RAW_QUANTIZATION,
            "quantization_bounds": [-100, 100], "geometry_required": True, "quantum_uv": 1,
            "clipping": False, "prompt_function": "evlm_mlx_train.reconstruction_prompt",
            "masked_target": "JSON integer array in masked-index order"}


def raw_eeg(start=0, channels=2):
    samples = 24 if start == 576 else 32
    return {"schema": ev.RAW_EEG_SCHEMA, "units": "uV", "synthetic": True,
            "geometry": {"shape": [channels, samples], "channel_names": ["EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF"][:channels],
                         "sample_rate_hz": 200, "quantum_uv": 1, "crop_start_sample": start},
            "values": [[(i + .2) * (1 if channel % 2 == 0 else -1) for i in range(samples)]
                       for channel in range(channels)]}


def write_tensor_file(path, tensors):
    header, data = {}, bytearray()
    for name, shape in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        size = count * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(data), len(data) + size]}
        data.extend(b"\x00" * size)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def fixture(root):
    base, adapter = root / "model", root / "source_adapter"
    base.mkdir()
    adapter.mkdir()
    config = {"model_type": "qwen3_5", "language_model_only": False,
              "text_config": {"num_hidden_layers": 2, "hidden_size": 4},
              "vision_config": {"out_hidden_size": 4}, "image_token_id": 20}
    module = "language_model.model.layers.1.mlp.down_proj"
    tensors = {module + ".weight": [4, 4], "vision_tower.proj.weight": [4, 4]}
    write_tensor_file(base / "model.safetensors", tensors)
    for name in ev.REQUIRED_ASSETS:
        (base / name).write_text("{}" if name.endswith(".json") else "fixture")
    (base / "config.json").write_text(json.dumps(config))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    adapter_cfg = {"fine_tune_type": "lora", "num_layers": 1,
                   "lora_parameters": {"rank": 2, "scale": 20, "dropout": 0, "keys": ["mlp.down_proj"]},
                   "model": str(base), "enable_thinking": False, "mask_prompt": True,
                   "loss_mask": "prefix_length <= target_position < token_length"}
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_cfg))
    write_tensor_file(adapter / "adapters.safetensors", {module + ".lora_a": [4, 2], module + ".lora_b": [2, 4]})
    return base, adapter, module, adapter_cfg


class EVLMContracts(unittest.TestCase):
    def test_fixed_serializer_zero_and_volts_golden(self):
        payload, meta = ev.encode_eeg(eeg())
        self.assertEqual(payload.splitlines()[2], "CP3 t:" + " ".join(["0"] * 28) + " p:-20 -20 -20 -20")
        self.assertEqual(len(payload.splitlines()), 10)
        self.assertEqual(meta["temporal_clipped_fraction"], 0)
        constant, _ = ev.encode_eeg(eeg(10e-6))
        self.assertEqual(constant.splitlines()[2], "CP3 t:" + " ".join(["2"] * 28) + " p:-20 -20 -20 -20")

    def test_reject_eeg_units_order_nonfinite_shape_and_target(self):
        bad = []
        x = eeg(); x["units"] = "uV"; bad.append(x)
        x = eeg(); x["channel_order"] = list(reversed(ev.CHANNELS)); bad.append(x)
        x = eeg(); x["values"][0][0] = float("nan"); bad.append(x)
        x = eeg(); x["values"][0].pop(); bad.append(x)
        x = eeg(); x["values"][0][0] = True; bad.append(x)
        x = eeg(); x["current_action"] = "mouse_click"; bad.append(x)
        x = eeg(); x["sample_rate_hz"] = 250; bad.append(x)
        x = eeg(); x["values"][0][0] = 10; bad.append(x)
        for value in bad:
            with self.subTest(value=list(value)):
                with self.assertRaises(ValueError):
                    ev.encode_eeg(value)

    def test_json_rejects_duplicate_keys_and_nonfinite(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                ev.strict_json(text)

    def test_action_schema_is_closed_and_dry_run(self):
        self.assertEqual(ev.parse_action('{"action":"click","x":10,"y":20}', 512, 320),
                         {"action": "click", "x": 10, "y": 20, "dry_run": True, "executed": False})
        bad = ['{"action":"click","x":512,"y":10}', '{"action":"click","x":true,"y":10}',
               '{"action":"click","x":10,"y":10,"execute":true}', '{"action":"shell","command":"echo hi"}',
               '```json\n{"action":"wait"}\n```', '{"action":"type","text":""}',
               '{"action":"wait","tool":"browser"}', 'mouse_click']
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    ev.parse_action(text, 512, 320)

    def test_prompt_rejects_special_tokens(self):
        with self.assertRaises(ValueError):
            ev.action_prompt('<|im_start|>system', {"width": 512, "height": 320}, "EEG")

    def test_raw32_geometry_channel_major_and_rounding(self):
        contract = ev.serialization_contract({"eeg_serialization": raw_descriptor()})
        document = raw_eeg()
        payload, metadata = ev.encode_eeg(document, contract)
        expected = list(range(32)) + [-i for i in range(32)]
        self.assertEqual(payload.splitlines()[0], "Quantization schema: " + ev.RAW_QUANTIZATION)
        self.assertEqual(payload.splitlines()[1], "Geometry: " + json.dumps(document["geometry"], sort_keys=True, separators=(",", ":")))
        self.assertEqual(payload.splitlines()[2], "Observed values (null=masked): " + json.dumps(expected, separators=(",", ":")))
        self.assertEqual(payload.splitlines()[3], "Masked indices: []")
        self.assertEqual(metadata["geometry"]["channel_names"], ["EEG F3-REF", "EEG F4-REF"])
        self.assertEqual(metadata["clipped_values"], 0)
        tail, tail_meta = ev.encode_eeg(raw_eeg(576), contract)
        self.assertEqual(tail_meta["geometry"]["shape"], [2, 24])
        self.assertIn('"crop_start_sample":576', tail)

    def test_raw_contract_and_geometry_mismatch_rejected(self):
        contract = ev.serialization_contract({"eeg_serialization": raw_descriptor()})
        with self.assertRaises(ValueError):
            ev.serialization_contract({"num_layers": 64})
        wrong = raw_descriptor(); wrong["clipping"] = True
        with self.assertRaises(ValueError):
            ev.serialization_contract({"eeg_serialization": wrong})
        bad = []
        doc = raw_eeg(); doc["units"] = "V"; bad.append(doc)
        doc = raw_eeg(); doc["geometry"]["crop_start_sample"] = 1; bad.append(doc)
        doc = raw_eeg(); doc["geometry"]["sample_rate_hz"] = 256; bad.append(doc)
        doc = raw_eeg(); doc["geometry"]["channel_names"] = ["F3", "F3"]; bad.append(doc)
        doc = raw_eeg(); doc["values"][0][0] = 100.6; bad.append(doc)
        doc = raw_eeg(); doc["values"][0][0] = None; bad.append(doc)
        doc = raw_eeg(); doc["target"] = [1, 2, 3, 4]; bad.append(doc)
        doc = raw_eeg(); doc["geometry"]["crop_start_sample"] = 576; bad.append(doc)
        for doc in bad:
            with self.assertRaises(ValueError):
                ev.encode_eeg(doc, contract)
        with self.assertRaises(ValueError):
            ev.encode_eeg(raw_eeg())
        with self.assertRaises(ValueError):
            ev.encode_eeg(eeg(), contract)

    def test_all_layers_adapter_scope_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, adapter, module, config = fixture(Path(tmp))
            audited = ev.audit_base(base)
            header = ev.tensor_header(adapter / "adapters.safetensors")
            earlier = module.replace("layers.1.", "layers.0.")
            audited["tensor_headers"][earlier + ".weight"] = deepcopy(audited["tensor_headers"][module + ".weight"])
            for suffix in ("lora_a", "lora_b"):
                header[earlier + "." + suffix] = deepcopy(header[module + "." + suffix])
            config["num_layers"] = 2
            config["eeg_serialization"] = raw_descriptor()
            contract = ev.validate_adapter_contract(config, header, audited)
            self.assertEqual(contract["modules"], sorted([earlier, module]))
            self.assertEqual(contract["serialization"]["schema"], ev.RAW_EEG_SCHEMA)

    def test_raw_adapter_bundle_selects_matching_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter, _, config = fixture(root)
            config["eeg_serialization"] = raw_descriptor()
            (adapter / "adapter_config.json").write_text(json.dumps(config))
            bundle = root / "bundle"
            manifest = ev.prepare_bundle(base, adapter, bundle, ev.sha256(adapter / "adapters.safetensors"), "a" * 40)
            document = ev.read_json(bundle / "synthetic_smoke/eeg.json")
            self.assertEqual(document["schema"], ev.RAW_EEG_SCHEMA)
            self.assertEqual(document["units"], "uV")
            self.assertEqual(document["geometry"]["sample_rate_hz"], 200)
            self.assertEqual(manifest["serializer"]["source_adapter_descriptor"], raw_descriptor())
            ev.encode_eeg(document, manifest["serializer"])
            ev.verify_bundle(bundle)

    def test_training_channel_cap_defaults_to_eight_and_rejects_invalid_metadata(self):
        old = ev.serialization_contract({"eeg_serialization": raw_descriptor()})
        self.assertEqual(old["max_channels_per_training_block"], 8)
        self.assertEqual(old["channel_count_max"], 8)
        ev.encode_eeg(raw_eeg(channels=8), old)
        for value in (True, False, 0, 9, 6.0, "6", None):
            descriptor = raw_descriptor()
            descriptor["max_channels_per_training_block"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                ev.serialization_contract({"eeg_serialization": descriptor})

    def test_six_channel_training_contract_rejects_wider_input_without_truncation(self):
        descriptor = raw_descriptor()
        descriptor["max_channels_per_training_block"] = 6
        contract = ev.serialization_contract({"eeg_serialization": descriptor})
        self.assertEqual(contract["channel_count_max"], 6)
        for channels in (1, 2, 6):
            _, metadata = ev.encode_eeg(raw_eeg(channels=channels), contract)
            self.assertEqual(metadata["geometry"]["shape"], [channels, 32])
            self.assertEqual(metadata["max_channels_per_training_block"], 6)
        for channels in (7, 8):
            with self.assertRaises(ValueError):
                ev.encode_eeg(raw_eeg(channels=channels), contract)

    def test_six_channel_bundle_fixture_and_manifest_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter, _, config = fixture(root)
            descriptor = raw_descriptor()
            descriptor["max_channels_per_training_block"] = 6
            config["eeg_serialization"] = descriptor
            (adapter / "adapter_config.json").write_text(json.dumps(config))
            bundle = root / "bundle"
            manifest = ev.prepare_bundle(base, adapter, bundle, ev.sha256(adapter / "adapters.safetensors"), "a" * 40)
            document = ev.read_json(bundle / "synthetic_smoke/eeg.json")
            self.assertEqual(document["geometry"]["shape"], [6, 32])
            self.assertEqual(len(document["geometry"]["channel_names"]), 6)
            self.assertEqual(len(document["values"]), 6)
            self.assertEqual(manifest["serializer"]["max_channels_per_training_block"], 6)
            self.assertEqual(manifest["serializer"]["source_adapter_descriptor"], descriptor)
            ev.encode_eeg(document, manifest["serializer"])
            ev.verify_bundle(bundle)
            manifest["serializer"]["max_channels_per_training_block"] = 8
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                ev.verify_bundle(bundle)

    def test_adapter_exact_targets_pair_rank_and_vision_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, adapter, module, config = fixture(Path(tmp))
            audited = ev.audit_base(base)
            header = ev.tensor_header(adapter / "adapters.safetensors")
            contract = ev.validate_adapter_contract(config, header, audited)
            self.assertEqual(contract["modules"], [module])
            self.assertEqual(contract["scale"], 20)
            for change in ("missing_pair", "wrong_shape", "vision", "missing_target", "wrong_layer"):
                changed, b = deepcopy(header), deepcopy(audited)
                current_config = deepcopy(config)
                if change == "missing_pair":
                    del changed[module + ".lora_b"]
                elif change == "wrong_shape":
                    changed[module + ".lora_b"]["shape"] = [4, 2]
                elif change == "vision":
                    changed["vision_tower.proj.lora_a"] = changed.pop(module + ".lora_a")
                elif change == "missing_target":
                    b["tensor_headers"][module.replace("down_proj", "gate_proj") + ".weight"] = b["tensor_headers"][module + ".weight"]
                    current_config["lora_parameters"]["keys"].append("mlp.gate_proj")
                else:
                    changed = {k.replace("layers.1.", "layers.0."): v for k, v in changed.items()}
                with self.subTest(change=change), self.assertRaises(ValueError):
                    ev.validate_adapter_contract(current_config, changed, b)

    def test_invalid_source_prompt_and_adapter_scale_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, adapter, _, config = fixture(Path(tmp))
            audited, header = ev.audit_base(base), ev.tensor_header(adapter / "adapters.safetensors")
            for field, value in (("scale", float("nan")), ("rank", True), ("dropout", 0.1)):
                changed = deepcopy(config)
                changed["lora_parameters"][field] = value
                with self.assertRaises(ValueError):
                    ev.validate_adapter_contract(changed, header, audited)
            config["enable_thinking"] = True
            with self.assertRaises(ValueError):
                ev.validate_adapter_contract(config, header, audited)

    def test_header_truncation_and_index_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truncated = root / "bad.safetensors"
            truncated.write_bytes(struct.pack("<Q", 900))
            with self.assertRaises(ValueError):
                ev.tensor_header(truncated)
            base, _, _, _ = fixture(root)
            (base / "model.safetensors.index.json").write_text('{"weight_map":{"missing":"model.safetensors"}}')
            with self.assertRaises(ValueError):
                ev.audit_base(base)

    def test_bundle_detects_weight_body_and_adapter_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter, _, _ = fixture(root)
            bundle = root / "bundle"
            result = ev.prepare_bundle(base, adapter, bundle, ev.sha256(adapter / "adapters.safetensors"), "a" * 40)
            self.assertFalse(result["action_execution"])
            self.assertFalse(result["portable_without_base"])
            ev.verify_bundle(bundle)
            path = base / "model.safetensors"
            original = path.read_bytes()
            path.write_bytes(original[:-1] + b"\x01")
            with self.assertRaisesRegex(ValueError, "base_shard_integrity"):
                ev.verify_bundle(bundle)
            path.write_bytes(original)
            path = bundle / "adapter/adapters.safetensors"
            path.write_bytes(path.read_bytes()[:-1] + b"\x01")
            with self.assertRaisesRegex(ValueError, "adapter_integrity"):
                ev.verify_bundle(bundle)

    def test_synthetic_screens_require_different_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "smoke"
            spec = ev.make_smoke(root)
            self.assertNotEqual(spec["cases"][0]["target_box"], spec["cases"][1]["target_box"])
            self.assertEqual(spec["cases"][0]["task"], spec["cases"][1]["task"])
            self.assertNotEqual(ev.sha256(root / "screen_0.png"), ev.sha256(root / "screen_1.png"))
            self.assertTrue(ev.read_json(root / "eeg.json")["synthetic"])
            self.assertEqual(ev.inspect_image(root / "screen_0.png")["width"], 512)

    def test_existing_gpu_lock_is_exclusive_and_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "gpu.lock"
            with self.assertRaises(FileNotFoundError):
                with ev.gpu_lock(lock):
                    pass
            lock.touch()
            with ev.gpu_lock(lock):
                with self.assertRaises(BlockingIOError):
                    with ev.gpu_lock(lock):
                        pass

    def test_wrong_existing_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "smac35/source_run/model"
            base.mkdir(parents=True)
            expected = root / "smac35/train_gpu.lock"
            expected.touch()
            wrong = root / "other.lock"
            wrong.touch()
            self.assertEqual(ev.canonical_gpu_lock(base, expected), expected.resolve())
            with self.assertRaises(ValueError):
                ev.canonical_gpu_lock(base, wrong)

    @unittest.skipUnless(importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_vlm") is not None,
                         "MLX-VLM is optional for pure CPU contract tests")
    def test_mlx_cpu_adapter_math_and_vision_identity(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        import numpy as np

        class Projection(nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = nn.Linear(4, 4, bias=False)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = Projection()

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [Layer(), Layer()]

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = Language()
                self.vision_tower = nn.Linear(4, 4)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = Model()
            module = "language_model.model.layers.1.mlp.down_proj"
            base = model.language_model.model.layers[1].mlp.down_proj
            x = mx.array([[1, 2, 3, 4]], dtype=mx.float32)
            a = mx.array([[.1, .2], [.3, .4], [.5, .6], [.7, .8]])
            b = mx.array([[.2, .3, .4, .5], [.6, .7, .8, .9]])
            expected = base(x) + (20 * ((x @ a) @ b)).astype(x.dtype)
            mx.eval(expected)
            mx.save_safetensors(str(root / "adapters.safetensors"), {module + ".lora_a": a, module + ".lora_b": b})
            before = {k: id(v) for k, v in tree_flatten(model.vision_tower.parameters())}
            receipt = ev.attach_adapter_strict(model, root, {"modules": [module], "rank": 2, "scale": 20})
            actual = model.language_model.model.layers[1].mlp.down_proj(x)
            mx.eval(actual)
            np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6)
            self.assertEqual(before, {k: id(v) for k, v in tree_flatten(model.vision_tower.parameters())})
            self.assertTrue(receipt["vision_unchanged"])
            self.assertTrue(str(mx.default_device()).startswith("Device(cpu"))


if __name__ == "__main__":
    unittest.main()
