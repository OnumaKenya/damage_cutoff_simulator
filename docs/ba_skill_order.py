# -*- coding: utf-8 -*-
"""
ブルーアーカイブ スキル順（開始スキル設定）探索スクリプト

モデル:
  - カードは全6枚。うち3枚が手札(スロット1,2,3)、3枚が山札(上から順)。
  - 初期配置(6枚の並び)が決定変数。
  - 手札のカードのみ使用可能。
  - スロット i のカードを使うと、そのカードは山札の一番下へ行き、
    山札の一番上のカードがスロット i にドローされる。

使い方: 下部の SKILLS / PLAN を書き換えて実行。
"""

from collections import deque
from itertools import permutations


# --- 手順の指定 -------------------------------------------------------------
# skill: 使いたいスキル名。None なら「何でもいい(捨て札/繋ぎ)」
# slot : 使ってほしいスロット番号(1-3)。None なら「どのスロットでもいい」
class Step:
    def __init__(self, skill=None, slot=None):
        self.skill = skill
        self.slot = slot

    def __repr__(self):
        s = self.skill if self.skill is not None else "*"
        p = self.slot if self.slot is not None else "*"
        return f"{s}@{p}"


def _use(hand, deck, idx):
    """スロット idx(0-2) のカードを使用した後の (hand, deck) を返す。"""
    new_hand = list(hand)
    new_deck = deque(deck)
    used = new_hand[idx]
    new_hand[idx] = new_deck.popleft()   # 山札トップをドロー
    new_deck.append(used)                # 使ったカードは山札の底へ
    return new_hand, new_deck


def _dfs(hand, deck, steps, i, trace, out):
    if i == len(steps):
        out.append(list(trace))
        return
    step = steps[i]
    slots = [step.slot - 1] if step.slot is not None else range(3)
    for idx in slots:
        if step.skill is not None and hand[idx] != step.skill:
            continue
        nh, nd = _use(hand, deck, idx)
        trace.append((idx + 1, hand[idx]))
        _dfs(nh, nd, steps, i + 1, trace, out)
        trace.pop()


# --- 手順間の制約 -----------------------------------------------------------
# trace は [(使用スロット, 使用スキル), ...] （PLAN と同じ添字）
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


def solve(skills, plan, constraints=()):
    """条件を満たす初期配置を列挙する。

    戻り値: [(初期配置tuple, [(使用スロット, 使用スキル), ...]), ...]
    初期配置tuple = (手札1, 手札2, 手札3, 山札上, 山札中, 山札下)
    """
    results = []
    for layout in permutations(skills):
        hand = list(layout[:3])
        deck = deque(layout[3:])
        traces = []
        _dfs(hand, deck, plan, 0, [], traces)
        for t in traces:
            if all(c(t) for c in constraints):
                results.append((layout, t))
    return results


def report(results, plan, limit=None):
    print(f"手順: {plan}")
    print(f"解の数: {len(results)}")
    print()
    shown = results if limit is None else results[:limit]
    for layout, trace in shown:
        hand = " / ".join(layout[:3])
        deck = " -> ".join(layout[3:])
        seq = "  ".join(f"[{s}]{c}" for s, c in trace)
        print(f"手札: {hand}   山札(上→下): {deck}")
        print(f"    使用順: {seq}")
    if limit is not None and len(results) > limit:
        print(f"... 他 {len(results) - limit} 件")


if __name__ == "__main__":
    # 6枚のスキル名（好きに書き換え）
    SKILLS = ["ハレ", "ヒカリ", "ノゾミ", "キサキ", "カヨコ", "ナギサ"]

    # 例: 1手目に A をスロット1で、2手目に B をスロット2で、
    #     3手目は何でもいいので1枚使い、4手目に C をスロット1で使いたい
    PLAN = [
        Step("ハレ"),
        Step("ヒカリ"),
        Step("ノゾミ"),
        Step("キサキ"),
        Step("カヨコ"),
        Step("ハレ"),
        Step("ナギサ"),
        Step("ヒカリ"),
        Step("カヨコ"),
        Step("ノゾミ"),
        Step("キサキ"),
        Step("ヒカリ"),
        Step("ハレ"),
        Step("ノゾミ"),
        Step("ナギサ"),
        Step("ヒカリ"),
        Step("キサキ"),
        Step("ノゾミ", 1),        
    ]

    # 手順間の制約（添字は PLAN の 0 始まり）
    CONSTRAINTS = [
        different_slots(2, 3),   # 3手目ノゾミ と 4手目キサキ を違うスロットに
    ]

    report(solve(SKILLS, PLAN, CONSTRAINTS), PLAN, limit=60)
