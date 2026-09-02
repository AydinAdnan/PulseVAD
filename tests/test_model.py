import torch
import pytest

from pulsevad.model import PulseVAD


def _layer_params(model, name):
    return sum(p.numel() for p in model._modules[name].parameters())


def test_exact_parameter_count():
    model = PulseVAD()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 81_090, f"Expected exactly 81,090 params, got {total_params}"
    assert trainable_params == 81_090, f"Expected all 81,090 params trainable, got {trainable_params}"


def test_layer_by_layer_parameter_table():
    model = PulseVAD()
    expected = {
        "adapter": 8_448,
        "conv0_dw": 1_408,
        "conv0_pw": 16_640,
        "block1": 8_320,
        "block2": 4_224,
        "block3": 14_848,
        "conv4_dw": 1_856,
        "conv4_pw": 8_448,
        "conv5": 16_640,
        "classifier": 258,
    }
    for name, count in expected.items():
        actual = _layer_params(model, name)
        assert actual == count, f"{name}: expected {count}, got {actual}"


def test_batchnorm_hyperparameters():
    # Spec rule #4: eps=1e-3 everywhere (PyTorch default 1e-5 shifts logits)
    for module in PulseVAD().modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            assert module.eps == 1e-3
            assert module.momentum == 0.1


def test_convolutions_are_bias_free():
    for module in PulseVAD().modules():
        if isinstance(module, torch.nn.Conv1d):
            assert module.bias is None


def test_forward_pass_shapes():
    model = PulseVAD()
    model.eval()
    x = torch.randn(8, 64, 21)

    logits = model(x, return_logits=True)
    assert logits.shape == (8, 2), f"Expected (8, 2), got {logits.shape}"

    probs = model(x, return_logits=False)
    assert probs.shape == (8,), f"Expected (8,), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_receptive_field_same_padding():
    # 21-frame input must stay 21 frames through every conv (dilation=2 on
    # conv4_dw with pad=28 keeps length constant)
    model = PulseVAD().eval()
    x = torch.randn(1, 64, 21)
    h = model.adapter(x)
    h = model.conv0_dw(h)
    h = model.conv0_pw(h)
    h = model.block1(h)
    h = model.block2(h)
    h = model.block3(h)
    h = model.conv4_dw(h)
    h = model.conv4_pw(h)
    h = model.conv5(h)
    assert h.shape == (1, 128, 21)


def test_backward_gradient_flow():
    model = PulseVAD()
    model.train()
    x = torch.randn(4, 64, 21)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
