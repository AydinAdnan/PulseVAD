"""Spec phase-05 gate: DepGraph student hits the 2.1k target and trains."""

import torch

from pulsevad.model import PulseVAD
from pulsevad.prune import PARAM_RANGE, PRUNE_PLAN, build_student, param_count


def test_student_param_count_in_gate():
    student = build_student()
    n = param_count(student)
    assert PARAM_RANGE[0] <= n <= PARAM_RANGE[1], f"{n} params, want {PARAM_RANGE}"
    assert n < 81_090 // 10  # actually ~97% smaller, not just "smaller"


def test_student_channel_targets():
    student = build_student()
    assert student.adapter.conv.out_channels == PRUNE_PLAN["adapter.conv"][1] == 12
    assert student.conv0_pw.conv.out_channels == 8
    assert student.block1.conv.out_channels == 8
    assert student.block2.conv.out_channels == 8
    # coupled layers were resized along with their group heads
    assert student.conv0_dw.out_channels == 12
    assert student.block3.subA_dw.out_channels == 8
    assert student.block3.skip.conv.out_channels == 8
    assert student.conv4_dw.out_channels == 8
    assert student.conv4_pw.conv.out_channels == 8
    assert student.conv5.conv.out_channels == 8
    assert student.classifier.in_features == 8
    assert student.classifier.out_features == 2  # head untouched


def test_student_forward_and_grad_flow():
    student = build_student()
    x = torch.randn(4, 64, 21)
    logits = student(x)
    assert logits.shape == (4, 2)
    logits.sum().backward()  # every surviving param must receive gradient
    assert all(p.grad is not None for p in student.parameters())


def test_teacher_weights_survive_pruning():
    """Pruning copies teacher weights into surviving channels.

    Uses the adapter (its 64-mel input is never pruned), so each surviving
    student filter must be an exact copy of some teacher filter.
    """
    torch.manual_seed(0)
    teacher = PulseVAD()
    student = build_student(teacher)
    kept = student.adapter.conv.weight  # (12, 64, 1)
    ref = teacher.adapter.conv.weight.detach()  # (128, 64, 1)
    assert kept.shape[0] == 12
    pool = ref.reshape(ref.shape[0], -1)
    for i in range(kept.shape[0]):
        row = kept[i].detach().reshape(-1)
        assert any(torch.equal(row, pool[j]) for j in range(pool.shape[0])), \
            f"student filter {i} not found in teacher"
