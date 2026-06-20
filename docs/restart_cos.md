# 多段リスタ最適化の係数空間(COS / Fang–Oosterlee)実装 — グリッド・求積なし

`app/backend/restart.py` はグリッド上の数値DP(`np.interp` 畳み込み + `np.trapezoid`
求積)で多段足切りを解く。`docs/cutoff.md` 付録 A が指摘するとおり、これは
**級数(COS 係数)空間で求積もグリッドもなしに厳密化できる**。本書はその係数空間版
`app/backend/restart_cos.py` の理論と実装対応をまとめる。`docs/cutoff.md` の付録
A.6(Bermudan 後ろ向き帰納)と A.7(積モデルへの読み替え)が数学的な正本であり、
本書は「何をどの式・どの関数で実装したか」の対応表である。

## 0. なぜグリッドが要らないのか(結論)

後ろ向き帰納
$$U_j(s)=\max\!\big(0,\,-g\,t_j+\mathbb E[U_{j+1}(s+T_j)]\big)$$
を構成する演算はすべて COS 係数で閉じる:

| 演算 | グリッド版(restart.py) | 係数空間版(restart_cos.py) |
|---|---|---|
| 畳み込み/期待値 $\mathbb E[\cdot(s+T_j)]$ | `np.interp` を512ノードで重ね合わせ | 余弦係数に CF $\varphi_{T_j}(u_k)$ を掛ける(畳み込み)/共役を掛ける(期待値) |
| 打ち切り $1\{s\ge d_j\}\cdot$ / $\max(0,\cdot)$ | `np.where` | F&O の $\mathcal C/\mathcal M$ 行列(本書 §3 の $I^{cc},I^{sc}$ 積分)。閉形式 |
| 関門 $d_j^*$ の決定 | grid 上で `cont>=0` の最初の点を走査 | 連続値 $\mathrm{cont}_j(s)$ を係数から直接評価し二分法で求根 |
| 通過率・成功率の積分 | `np.trapezoid` | 余弦係数と $\psi^c_k,\psi^s_k$ の内積(閉形式) |

誤差は**級数打ち切りのみ**(求積誤差ゼロ)。`restart.py` で起きた「通過率>100%」
(序盤の幅の狭いセグメントを粗グリッドで台形積分して質量が膨らむ)は、係数空間では
**原理的に起こらない**。

## 1. 表現

共通区間 $[a,b]$ 上で関数を**半区間余弦係数**で表す($\theta=s-a$):
$$g(s)=\sum_{k=0}^{N-1}\tilde c_k\cos(u_k\theta),\qquad u_k=\frac{k\pi}{L},\ L=b-a,\quad
\tilde c_0=\frac1L\!\int_a^b g,\ \ \tilde c_k=\frac2L\!\int_a^b g\cos(u_k\theta)\,ds\ (k\ge1).$$
($\tilde c_0$ は DC を半分にした「チルダ」規約。`cos.py` の `Fk` と同じ。)

右端 $b=\sum_j \overline{T_j}$(各セグメント増分の台上限の和=累積ダメージの最大)。
**左端 $a<0$ のマージン**が重要: 半区間余弦は偶対称周期拡張(周期 $2L$)なので、
$s=0$ 付近に集中した密度を素朴に畳み込むと、$s<0$ 側のミラー像が増分シフトで台に
折り込み**質量が二重化**する(実装初期に S₁₂ の質量が 2.0 になった)。各畳み込みで
「ミラー + 増分」が $[a,b]$ の外(左)に収まるよう、$a=-1.05\max_j\overline{T_j}$
(最大セグメント増分ぶん負側へ)とすると二重化は消える(検証で S₂₈ の超過確率が
厳密値 5.343% と一致)。これは F&O が窓 $[a,b]$ を台より広く取るのと同じ理屈。

畳み込みの途中で正弦成分が出るため、中間状態は **(余弦係数 $C_k$, 正弦係数 $S_k$)** の
対で持つ。打ち切り(`truncate`)の**出力は必ず半区間余弦のみ**(切断後の関数は
半区間余弦基底で完全に表せる)。畳み込み(`convolve`)が余弦のみ→(余弦,正弦)に戻す。

## 2. 各演算(`restart_cos.py` の関数対応)

記号: $\varphi_k=\varphi_T(u_k)$(セグメント増分の特性関数)。

- **密度の初期係数** `_density_coeffs(seg)`:
  $\tilde c_k=\frac2b\mathrm{Re}\,\varphi_k$($k\ge1$), $\tilde c_0=\frac1b$, $S_k=0$。
- **積分** `_integrate(C,S,d)` $=\int_d^b g$:
  $\sum_k\!\big[C_k\,\psi^c_k(d)+S_k\,\psi^s_k(d)\big]$,
  $\psi^c_0=b-d,\ \psi^c_k=-\sin(u_kd)/u_k,\ \psi^s_0=0,\ \psi^s_k=(\cos(u_kd)-(-1)^k)/u_k$。
  (通過率 $p_j=\int_{d_j}^b f_j$、成功率 $=\int_D^b (f_K^{tr}*T_K)$。)
- **打ち切り** `_truncate(C,S,d)` $\to$ 余弦のみ $\tilde c'_m$:
  $$\tilde c'_m=\mathrm{pref}_m\Big[\underbrace{\tfrac12\big(\mathrm{TE}(C)+\mathrm{HC}(C)\big)_m}_{\sum_k C_k I^{cc}_{m,k}}
  +\underbrace{\tfrac12\big(\mathrm{HS}(S)+\mathrm{TO}(S)\big)_m}_{\sum_k S_k I^{sc}_{m,k}}\Big],$$
  $\mathrm{pref}_0=1/b,\ \mathrm{pref}_{m\ge1}=2/b$。基本積分($[d,b]$ 上)は
  $$I^{cc}_{m,k}=\tfrac12\big(\mathrm{SC}_{|k-m|}+\mathrm{SC}_{k+m}\big),\quad
    I^{sc}_{m,k}=\tfrac12\big(\mathrm{SS}_{k+m}+\mathrm{sgn}(k-m)\,\mathrm{SS}_{|k-m|}\big),$$
  $$\mathrm{SC}_r=\int_d^b\!\cos(u_r s)ds=\begin{cases}b-d&r=0\\-\sin(u_rd)/u_r&r\ge1\end{cases},\quad
    \mathrm{SS}_r=\int_d^b\!\sin(u_r s)ds=\begin{cases}0&r=0\\(\cos(u_rd)-(-1)^r)/u_r&r\ge1\end{cases}.$$
  TE/HC は SC を核とする Toeplitz(偶)/Hankel 行列ベクトル積、TO/HS は SS(奇/Hankel)。
  いずれも `_corr_valid`(FFT 相互相関)で $O(N\log N)$。$\psi^c_k=\mathrm{SC}_k,\psi^s_k=\mathrm{SS}_k$。
- **畳み込み**(密度、前向き)`convolve`: 余弦のみ $\tilde c$ から
  $C'_k=\tilde c_k\,\mathrm{Re}\,\varphi_k,\ S'_k=\tilde c_k\,\mathrm{Im}\,\varphi_k$($g*T$)。
- **期待値**(価値、後ろ向き)`expect`: 余弦のみ $A$ から
  $\mathrm{contCos}_k=A_k\mathrm{Re}\,\varphi_k,\ \mathrm{contSin}_k=-A_k\mathrm{Im}\,\varphi_k$
  ($\mathbb E[V(s+T)]$、= 共役を掛ける)。
- **評価** `_eval(C,S,s)` $=\sum_k C_k\cos(u_ks)+S_k\sin(u_ks)$(関門の求根に使用)。

## 3. アルゴリズム

- **後ろ向き帰納**(固定 $g$, `_backward`): 終端 $V_{K+1}=1\{s\ge D\}$(余弦係数は
  `_indicator_coeffs`)。各段 $j$ で `expect`→DC を $-g t_j$ シフト→`_find_gate`(二分法で
  $\mathrm{cont}_j(s)=0$)→`_truncate` で $V_j$。開始値 $V_0(g)=-g t_0+\mathbb E[V_1(T_0)]$。
- **Dinkelbach**(`_optimize`): $V_0(g)=0$ を二分法で解き $g^*$。$g^*$ で 1 回掃いて関門 $d_j^*$。
- **前向き**(`forward_metrics`): 関門固定で通過率・成功率・期待時間を §2 の積分で算出。
- **積モデル**(`analyze_product`): $G=-\sum\ln Y$ 座標(高ダメージ=大 $G$)で和モデルと
  同型。セグメント増分 CF は $\varphi_G(u)=\overline{\varphi_{S}(u)}$、目標は
  $D_{\mathrm{thr}}=-\ln\frac{\tilde H_1-D}{\tilde H_1}$、関門は $d=\tilde H_1(1-e^{-g})$ で
  ダメージに戻す(`docs/cutoff.md` A.7)。

## 4. 数値の注意

- **項数 $N$**: 最も狭いセグメント増分の台幅 $w_{\min}$ を解像するよう
  $N\approx\mathrm{clip}(b/w_{\min}\cdot 12,\ 1024,\ 8192)$ で自動決定。価値関数は関門で
  折れる($C^0$)ため係数は $O(m^{-2})$ で減衰、中程度の $N$ で十分。
- **共鳴** $u_k=u_m$($k=m$, SC/SS の $r=0$)は $\int_d^b\cos/\sin$ の極限値で別扱い。
- **裾のクランプ**: COS の CDF は裾で僅かに $[0,1]$ を外れうるため、通過率・成功率は
  $[0,1]$ にクランプする(`restart.py` と同じ安全弁)。
- **検証**: `tests/test_restart_cos.py` で (a) `_corr_valid` を素朴な相関と一致、
  (b) 前向き通過率・成功率を MC と一致、(c) 関門・スループットを `restart.py`
  グリッド版と一致(粗グリッド由来の差を除く)を確認する。

## 5. 区間ごとのダメージと独立な成功確率

各区間 $j$ に「ダメージとは独立な成功確率 $q_j$」(その区間を回しきって次へ進める
確率。失敗すると**末尾でリスタート**し、所要時間 $t_j$ は消費済み)を付けられる。
リスタート価値は更新報酬の基準で 0 なので、続行価値の期待値項に $q_j$ が掛かるだけ:
$$\mathrm{cont}_j(s)=-g\,t_j+q_j\,\mathbb E[V_{j+1}(s+T_j)].$$
$q_j$ は状態 $s$ に依らない定数なので係数空間でもスカラー倍で閉じる
(`restart_cos._backward` / グリッド版 `restart._backward`)。終端は $V_{K+1}=1\{s\ge D\}$ で不変、
開始は $V_0(g)=-g\,t_0+q_0\,\mathbb E[V_1(T_0)]$。$q_j<1$ ほど続行価値が割引かれ関門 $d_j$ は
上がる(続行が不利→より高いダメージを要求)。

前向き(`forward_metrics`)では、区間 $j$ に到達して回す確率は
$\big(\prod_{i<j}q_i\big)\cdot p_j^{\mathrm{dmg}}$(ダメージ関門の累積通過率に独立成功積を掛けたもの)。
$\;$期待時間 $\;\mathbb E[T]=t_0+\sum_{j\ge1}t_j\big(\prod_{i<j}q_i\big)p_j^{\mathrm{dmg}}$、$\;$完走確率は最終区間ぶんも
含め $\prod_{i\le K}q_i$ を掛ける。`baseline_nogate` も独立失敗のリスタートを反映して
$\mathbb E[T]=\sum_j t_j\prod_{i<j}q_i$, $\;\mathrm{success}=\big(\prod_i q_i\big)\,P(\text{合計}\ge D)$。
全 $q_j=1$ なら独立確率なしの素の DP に完全一致する(`test_seg_success_identity_when_all_one`)。
MC 照合は `test_seg_success_matches_montecarlo`。UI は区間カードの「成功率%」入力
(`restart-seg-success-store`, 区間開始境界キー)。

## 6. 既知の限界 / 今後

- 打ち切り行列ベクトル積は FFT 相互相関で $O(N\log N)$。付録 A.4 の Toeplitz+Hankel
  を陽に組む形(さらに係数結合を $D$ 走査で前計算)にすれば、目標 $D$ を動かす用途で
  もう一段速くできる。
- 原子(回避の点質量)は CF にそのまま含めて扱う(係数空間の積分は原子を
  正しく取り込む。Gibbs は点値評価でのみ問題で、本実装は常に積分なので影響なし)。
