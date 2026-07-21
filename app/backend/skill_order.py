# -*- coding: utf-8 -*-
"""ブルーアーカイブ スキル順(開始スキル設定)探索ロジック。

モデル:
  - カードは全6枚。うち3枚が手札(スロット1,2,3)、3枚が山札(上から順)。
  - 初期配置(6枚の並び)が決定変数。
  - 手札のカードのみ使用可能。
  - 通常カードをスロット i で使うと、そのカードは山札の一番下へ行き、
    山札の一番上のカードがスロット i にドローされる。

拡張:
  - 複製スキル: 複製キャラ A が対象 B を指定して撃つと、カード A が
    その場で「B(コピー)」に変化する(山札への移動・ドローは発生しない)。
    「B(コピー)」を使用するとカードは A に戻って山札の一番下へ行く。
  - ドローフラグ: 手順ステップにドローフラグが付いている場合、
    使用カードが山札の下へ行った後、「使用したスキルの元カード」が
    山札にあるかチェックし、あれば元々カードがあったスロットへ
    そのカードがドローされる(山札の途中からでも引き抜く)。
    無ければ通常どおり山札の一番上をドローする。
    - 通常カード A の場合: A 自身が山札に落ちた直後なので必ず手札に戻る。
    - B(コピー) (複製キャラ A が B を複製したカード) の場合: カードは A に
      戻って山札の下へ行き、元の B カードが山札にあれば引き抜かれる。

カード表現:
  - 通常カード: ("N", i)          i はスキル添字 (0..5)
  - コピーカード: ("C", a, b)      a=複製キャラ添字, b=複製対象添字
"""

from itertools import permutations


class SearchBudgetExceeded(Exception):
    """探索ノード数が上限を超えた(ワイルドカード過多などで組合せ爆発)。"""


class _StopSearch(Exception):
    """十分な数の解が集まったので探索を打ち切る(内部用)。"""


class Step:
    """手順の1ステップ。

    skill      : 使うスキルの添字 (0..5)。None なら「何でもいい(繋ぎ)」。
                 ただしワイルドカードは複製キャラの元カードを使わない
                 (複製対象を決められないため)。
    use_copy   : True ならスキル skill のコピーカード「skill(コピー)」を使う。
    copy_target: skill が複製キャラのとき必須。複製する対象スキルの添字。
    slot       : 使ってほしいスロット番号(1-3)。None なら任意。
    draw       : ドローフラグ。
    """

    def __init__(self, skill=None, *, use_copy=False, copy_target=None,
                 slot=None, draw=False):
        self.skill = skill
        self.use_copy = use_copy
        self.copy_target = copy_target
        self.slot = slot
        self.draw = draw

    def __repr__(self):
        s = "*" if self.skill is None else (
            f"copy({self.skill})" if self.use_copy else str(self.skill))
        p = self.slot if self.slot is not None else "*"
        d = "+draw" if self.draw else ""
        return f"{s}@{p}{d}"


def card_label(card, names):
    """カードの表示名。"""
    if card[0] == "N":
        return names[card[1]]
    return f"{names[card[2]]}(コピー)"


def _matches(card, step, copiers):
    if step.skill is None:
        # ワイルドカード: 複製キャラの元カードは対象外(複製対象が不定のため)
        return not (card[0] == "N" and card[1] in copiers)
    if step.use_copy:
        return card[0] == "C" and card[2] == step.skill
    return card[0] == "N" and card[1] == step.skill


def _use(hand, deck, idx, step, copiers):
    """スロット idx のカードを使用した後の (hand, deck, action_label_key) を返す。

    action_label_key: トレース表示用 ("transform" | "use")。
    """
    card = hand[idx]
    nh = list(hand)
    nd = list(deck)

    # 複製キャラの元カード → その場でコピーカードに変化(山札は動かない)
    if card[0] == "N" and card[1] in copiers:
        nh[idx] = ("C", card[1], step.copy_target)
        return nh, nd, "transform"

    if card[0] == "C":
        # コピー使用: 複製キャラのカードに戻って山札の底へ
        nd.append(("N", card[1]))
        origin = ("N", card[2])   # ドローフラグで探す「元カード」
    else:
        nd.append(card)
        origin = card

    if step.draw and origin in nd:
        nd.remove(origin)         # 山札の途中からでも引き抜く
        nh[idx] = origin
    else:
        nh[idx] = nd.pop(0)       # 通常ドロー(山札トップ)
    return nh, nd, "use"


def _dfs(hand, deck, steps, i, copiers, trace, out, budget):
    if i == len(steps):
        out.append(list(trace))
        return
    step = steps[i]
    slots = [step.slot - 1] if step.slot is not None else range(3)
    for idx in slots:
        if idx < 0 or idx > 2:
            continue
        if not _matches(hand[idx], step, copiers):
            continue
        budget[0] -= 1
        if budget[0] < 0:
            raise SearchBudgetExceeded
        nh, nd, action = _use(hand, deck, idx, step, copiers)
        trace.append((idx + 1, hand[idx], action, step))
        _dfs(nh, nd, steps, i + 1, copiers, trace, out, budget)
        trace.pop()


# --- 手順間の制約 -----------------------------------------------------------
def different_slots(*indices):
    """指定した手順どうしを全て違うスロットにする制約。添字は0始まり。"""
    def check(trace):
        slots = [trace[i][0] for i in indices]
        return len(set(slots)) == len(slots)
    return check


def same_slot(*indices):
    """指定した手順どうしを全て同じスロットにする制約。添字は0始まり。"""
    def check(trace):
        slots = [trace[i][0] for i in indices]
        return len(set(slots)) == 1
    return check


def solve(n_skills, copiers, plan, constraints=(), *,
          max_results=None, node_budget=2_000_000):
    """条件を満たす初期配置を列挙する。

    n_skills   : スキル枚数(通常6)。
    copiers    : 複製キャラの添字集合。
    plan       : Step のリスト。
    constraints: trace を受け取り bool を返す関数のリスト。
    max_results: 解がこの数に達したら探索を打ち切る(None なら無制限)。
    node_budget: 探索ノード数の上限。超えたら SearchBudgetExceeded。

    戻り値: (results, truncated)
    results   = [(初期配置tuple, trace), ...]
    truncated = max_results により打ち切った場合 True
    初期配置tuple = (手札1, 手札2, 手札3, 山札上, 山札中, 山札下) ※スキル添字
    trace = [(使用スロット1-3, 使用カード, action, step), ...]
    """
    copiers = set(copiers)
    budget = [node_budget]
    results = []
    truncated = False
    try:
        for layout in permutations(range(n_skills)):
            hand = [("N", i) for i in layout[:3]]
            deck = [("N", i) for i in layout[3:]]
            traces = []
            _dfs(hand, deck, plan, 0, copiers, [], traces, budget)
            for t in traces:
                if all(c(t) for c in constraints):
                    results.append((layout, t))
                    if max_results is not None and len(results) >= max_results:
                        raise _StopSearch
    except _StopSearch:
        truncated = True
    return results, truncated


SLOT_LABELS = ("左", "中", "右")


def trace_entry_label(entry, names):
    """トレース1要素の表示文字列 ([左/中/右]表記)。"""
    slot, card, action, step = entry
    pos = SLOT_LABELS[slot - 1]
    if action == "transform":
        return f"[{pos}]{names[card[1]]}→{names[step.copy_target]}(コピー)"
    label = card_label(card, names)
    if step.draw:
        label += "(ドロー)"
    return f"[{pos}]{label}"
