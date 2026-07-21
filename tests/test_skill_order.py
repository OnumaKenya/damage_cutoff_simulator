# -*- coding: utf-8 -*-
"""スキル順探索 (app.backend.skill_order) のテスト。"""

from app.backend.skill_order import (
    SearchBudgetExceeded,
    Step,
    different_slots,
    solve,
    trace_entry_label,
)

NAMES = ["A", "B", "C", "D", "E", "F"]


def test_basic_plan_has_solutions():
    plan = [Step(0), Step(1), Step(2), Step(3), Step(0, slot=1)]
    res, truncated = solve(6, set(), plan)
    assert res and not truncated


def test_copy_transform_and_use():
    # A(=0, 複製) が B(=1) を複製 → B(コピー) を使用
    plan = [Step(0, copy_target=1), Step(1, use_copy=True)]
    res, _ = solve(6, {0}, plan)
    assert res
    layout, trace = res[0]
    # 変化はその場で起きるので、コピー使用は同じスロット
    assert trace[0][0] == trace[1][0]
    assert trace[0][2] == "transform"
    assert trace[1][1] == ("C", 0, 1)
    labels = [trace_entry_label(e, NAMES) for e in trace]
    assert labels[0].endswith("A→B(コピー)")
    assert labels[1].endswith("B(コピー)")


def test_draw_flag_returns_card_to_hand():
    # ドローフラグ付きで使うと自分のカードが手札に戻るため連打できる
    res, _ = solve(6, set(), [Step(1, draw=True), Step(1)])
    assert res
    for _, trace in res:
        assert trace[0][0] == trace[1][0]  # 同じスロットで連打

    # フラグ無しでは同じカードは2連打できない
    res_nodraw, _ = solve(6, set(), [Step(1), Step(1)])
    assert not res_nodraw


def test_draw_flag_on_copy_pulls_original_from_deck():
    # B(=1, 複製) が A(=0) を複製 → A(コピー) をドロー付きで使用 →
    # B に戻って山札の底へ行き、山札に A があれば同じスロットへ引き抜かれる
    plan = [
        Step(1, copy_target=0),
        Step(0, use_copy=True, draw=True),
        Step(0),
    ]
    res, _ = solve(6, {1}, plan)
    assert res
    checked = False
    for layout, trace in res:
        if 0 in layout[3:]:  # 元の A が初期山札にいる配置
            checked = True
            assert trace[1][0] == trace[2][0]
    assert checked


def test_constraints():
    res, _ = solve(6, set(), [Step(None), Step(None)],
                   [different_slots(0, 1)])
    assert res
    assert all(t[0][0] != t[1][0] for _, t in res)


def test_wildcard_skips_copier_original():
    res, _ = solve(6, {0}, [Step(None)])
    assert res
    assert all(t[0][1] != ("N", 0) for _, t in res)


def test_max_results_truncation():
    res, truncated = solve(6, set(), [Step(None)], max_results=10)
    assert truncated and len(res) == 10


def test_node_budget():
    plan = [Step(None)] * 10
    try:
        solve(6, set(), plan, node_budget=1000)
    except SearchBudgetExceeded:
        pass
    else:
        raise AssertionError("budget should be exceeded")
