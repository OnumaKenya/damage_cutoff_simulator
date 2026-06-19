# 積モデル: HP 依存ダメージの対数変換と COS 法

## 1. 問題設定

`docs/edge.md`・`docs/saddlepoint.md` では、独立な確率変数の**和** $S_n = \sum_i X_i$ の分布を Edgeworth 展開・サドルポイント近似・COS 法で計算した。そこでは「ターゲットの残り HP でダメージ倍率が変わる」ミカやフィーナのようなケースは、ダメージが前の Hit の結果に依存するため**対象外**としていた。

本稿では、まさにその HP 依存ダメージを扱う。鍵は、HP 依存の累積ダメージが独立な確率変数の**積**として書ける、という観察である。積は対数を取れば和に戻るので、和に対して築いた道具(特性関数の COS 反転)がそのまま使える。これは [Qiita の記事 (abira)](https://qiita.com/abira/items/eb93a1b9d6eea5911b99) が FFT で行った計算を、特性関数の直接反転に置き換えたものに相当する。

一言でいえば、本稿のテーマは**「積で表される分布」を対数で和に直して扱う**ことである。

## 2. HP 依存ダメージが積になること

ミカのように、与ダメージが敵の現在 HP に比例して変化する倍率を持つ場合を考える。記法を以下のように置く。

- $H$ : 敵の最大 HP(倍率の基準)
- $H_1$ : 攻撃開始時の敵 HP
- $H_n$ : $n$ 回目の攻撃**直前**の敵 HP
- $R_1$ : HP 満タン時($H_n = H$)の倍率、$R_0$ : HP=0 時の倍率、$\Delta R := R_1 - R_0$
- $x_n$ : $n$ 回目の**基礎ダメージ**(会心・非会心・miss の一様分布混合。`docs/edge.md` の 1 Hit ダメージそのもの)

倍率は現在 HP $H_n$ に線形で依存する。$\beta := \Delta R / H$ とおくと、

$$
\text{倍率}(H_n) = \frac{\Delta R}{H} H_n + R_0 = \beta H_n + R_0
$$

で、$H_n = H$ なら $\Delta R + R_0 = R_1$、$H_n = 0$ なら $R_0$ と両端を正しく再現する。1 Hit の実ダメージと HP の漸化式は

$$
D_n = (\beta H_n + R_0)\, x_n, \qquad H_{n+1} = H_n - D_n
$$

である。このままでは $H_{n+1}$ が $H_n$ に非線形(乗法的)に依存し、和の理論が使えない。ここで**シフトした HP** $\tilde H_n := H_n + R_0/\beta = H_n + (R_0/\Delta R)H$ を導入すると、漸化式が純粋な乗法になる。

$$
\tilde H_{n+1} = H_{n+1} + \frac{R_0}{\beta} = (\tilde H_n - \tfrac{R_0}{\beta})(1 - \beta x_n) - R_0 x_n + \frac{R_0}{\beta} = \tilde H_n\,(1 - \beta x_n)
$$

(展開すると $R_0 x_n$ の項が打ち消し合う。)よって $N$ 回攻撃後は

$$
\tilde H_{N+1} = \tilde H_1 \prod_{n=1}^N (1 - \beta x_n)
$$

となり、累積ダメージ $D := H_1 - H_{N+1} = \tilde H_1 - \tilde H_{N+1}$ は**独立な確率変数の積**で書ける。

$$
\boxed{\,D = \tilde H_1\left(1 - \prod_{n=1}^N (1 - \beta x_n)\right), \qquad \tilde H_1 = H_1 + \frac{R_0}{\Delta R}H\,}
$$

各 Hit が独立な基礎ダメージ $x_n$ を持つので、$Y_n := 1 - \beta x_n$ は独立で、$D$ は積 $P := \prod_n Y_n$ の単調関数である。$x_n \ge 0$ かつ $\beta x_n < 1$(HP が正)である限り $Y_n \in (0, 1]$、$P \in (0,1]$ なので、$D \in [0, \tilde H_1)$ に収まる。

## 3. 積を対数で和に直す

積の分布は直接には扱いにくいが、対数を取れば独立和に戻る。

$$
S := \ln P = \sum_{n=1}^N \ln Y_n
$$

$S$ は独立な確率変数 $\ln Y_n$ の和なので、特性関数は積に分解される。これが本稿の要で、`docs/saddlepoint.md` の COS 法を $S$ に対してそのまま適用できる理由である。

$$
\varphi_S(u) = \prod_{n=1}^N \varphi_{\ln Y_n}(u)
$$

(Qiita の記事の変数 $\lambda_n$ は $\lambda_n = -\frac{1}{\beta}\ln Y_n$、$\Lambda = \sum_n\lambda_n = -S/\beta$ という $\ln Y_n$ のアフィン変換にすぎない。記事はこの $\Lambda$ の分布を FFT で畳み込むが、ここでは特性関数を直接反転する。)

## 4. $\ln Y$ の特性関数(対数一様 = 切断指数分布)

基礎ダメージ $x$ が一様成分 $U(a, b)$ を持つとき、$Y = 1 - \beta x$ はアフィン変換なので $Y \sim U(p, q)$、$p = 1 - \beta b$、$q = 1 - \beta a$($b > a$ より $p < q$、ともに正)である。その対数 $s = \ln Y$ の密度は、$Y = e^s$、$dY/ds = e^s$ から

$$
f_s(s) = f_Y(e^s)\,e^s = \frac{e^s}{q - p}, \qquad s \in [\ln p,\ \ln q]
$$

となる。これは指数的に増大する密度を持つ**対数一様分布**(= 切断指数分布)である。一様分布が COS 法で $\mathrm{sinc}$ 型の特性関数を持ったのに対し、対数を取るとこの形になる。特性関数は閉形式で

$$
\varphi_{\ln Y}(u) = \int_{\ln p}^{\ln q} e^{ius}\,\frac{e^s}{q-p}\,ds = \frac{1}{q-p}\,\frac{e^{(1+iu)\ln q} - e^{(1+iu)\ln p}}{1 + iu} = \frac{q\,e^{iu\ln q} - p\,e^{iu\ln p}}{(q - p)(1 + iu)}
$$

である。分母の $1 + iu$ は 0 にならず数値的に安定で、$u = 0$ で $(q-p)/((q-p)\cdot 1) = 1$ と総質量を正しく返す。miss などの 1 点分布($x = c$ で退化、$Y = 1 - \beta c$)は点質量 $\varphi(u) = e^{iu\ln(1-\beta c)}$ である。

1 Hit が複数成分の混合(重み $w_j$)なら和

$$
\varphi_{\ln Y}(u) = \sum_j w_j\,\varphi_{\ln Y_j}(u)
$$

を取り、$N$ Hit の和 $S$ は独立性から積 $\varphi_S(u) = \prod_n \varphi_{\ln Y_n}(u)$ になる(同分布 $N$ Hit なら $\varphi_{\ln Y}(u)^N$)。

## 5. COS 法による $S$ の反転と $D$ への写像

$S = \ln P$ は**有界な台** $[A, B]$ を持つ。各 Hit の $\ln Y_n$ は $[\ln(\min_j p_j),\ \ln(\max_j q_j)]$ に収まるので、

$$
A = \sum_{n} \ln\big(\min_j p_{n,j}\big), \qquad B = \sum_{n} \ln\big(\max_j q_{n,j}\big) \ (\le 0)
$$

である。さらに $N \ge 2$ では和の密度が台の端で多項式的に 0 に落ちるため、台 $[A, B]$ をそのまま COS の区間に取れば Gibbs 振動はほぼ出ない(`docs/saddlepoint.md` のキュムラント窓クリップは、$N$ が大きく台幅が $\sigma$ スケールより広がる場合の項数削減策で、必要なら同様に併用できる)。COS 展開は

$$
f_S(s) \approx {\sum_{k=0}^{N_t-1}}' F_k \cos\big(u_k(s - A)\big), \quad u_k = \frac{k\pi}{B - A}, \quad F_k = \frac{2}{B-A}\,\mathrm{Re}\!\left[\varphi_S(u_k)\,e^{-iu_k A}\right]
$$

で、項別積分して $F_S(s) = \tfrac{F_0}{2}(s-A) + \sum_{k\ge1} F_k \frac{\sin(u_k(s-A))}{u_k}$ を得る(プライムは $k=0$ を半分にする記法)。

最後に $D$ へ写す。$D = \tilde H_1(1 - e^S)$ は $S$ について**単調減少**である($S$ が 0 に近いほど積 $P$ が 1 に近く、ダメージは小さい)。$S(D) = \ln(1 - D/\tilde H_1)$ とおくと、

$$
F_D(D) = P(D' \le D) = P\big(S \ge S(D)\big) = 1 - F_S\big(S(D)\big)
$$

$$
f_D(D) = f_S\big(S(D)\big)\,\left|\frac{dS}{dD}\right| = \frac{f_S\big(S(D)\big)}{\tilde H_1 - D}
$$

となる(符号の向きに注意。密度は $|dS/dD|$ なので向きに依らないが、CDF は反転する)。

## 6. 誤差解析(構築した $S$ が指数の肩に乗ること)

最後に $D = \tilde H_1(1 - e^S)$ で $S$ を**指数の肩**に乗せて $D$ へ写すため、$S$ 側の誤差がどう伝播するかを押さえておく。結論は**「累積分布(裾確率)の誤差は指数で増幅されないが、密度の絶対誤差は $e^{-S}$ 倍されて高ダメージ側で指数的に膨らむ」**である。

### 6.1 $S = \ln P$ 側の誤差源

COS 法の誤差は通常 (i) 積分範囲の打ち切り、(ii) 余弦係数 $F_k$ の求積、(iii) 級数打ち切り、に分かれるが、本手法では (i)(ii) が**消える**。

- **範囲打ち切り誤差 = 0**: $S$ はコンパクト台 $[A,B]$ を持ち、その台をそのまま COS 区間に使うので $\varphi_S(u_k) = \int_A^B e^{iu_k s} f_S(s)\,ds$ が厳密に成り立ち、$F_k$ は真のフーリエ余弦係数と一致する(無限台を切る誤差が無い)。
- **求積誤差 = 0**: $\varphi_S = \prod_n \varphi_{\ln Y_n}$ は閉形式で、数値積分を一切経ない。

したがって連続部に残るのは**級数打ち切り誤差だけ**である。$f_{\ln Y}(s) = e^s/(q-p)$ は端点で跳びを持つので、その $N$ 重畳み込み $f_S$ は $C^{N-2}$ 級で、係数包絡線は $F_k = O(k^{-(N-1)})$ 程度。打ち切り誤差は

$$
\varepsilon_f^{S}(N_t) = O\!\big(N_t^{-(N-1)}\big), \qquad \varepsilon_F^{S}(N_t) = O\!\big(N_t^{-N}\big)
$$

(CDF は項別積分で 1 次得をする)で、Hit 数 $N$ が増えるほど速く落ちる。なお原子分離(DP)は、純原子部の CF が $u\to\infty$ で減衰せず Gibbs を生むのを防ぐ手当てで、これを怠ると後述の指数増幅でその振動が高ダメージ側で爆発する。

### 6.2 指数写像による誤差伝播

$s = S(D) = \ln(1 - D/\tilde H_1)$ は決定論的な単調全単射なので、誤差は**確率(CDF)**と**密度(PDF)**で全く違う振る舞いをする。

**(a) 累積分布・裾確率 — 指数で増幅されない。**

$$
F_D(D) = 1 - F_S\big(S(D)\big) \quad\Longrightarrow\quad \big|\Delta F_D(D)\big| = \big|\Delta F_S(s)\big|.
$$

確率質量は単調な座標付け替えで保存されるので、$S$ 側の CDF 絶対誤差がそのまま移るだけ。**裾確率 $P(D>x) = F_S(S(x))$ の絶対誤差は指数の肩に乗っても膨らまない。** 精度を測るならこちらが頑健である(ただし「絶対誤差は不変でも、$P\sim10^{-8}$ に対し誤差床が $10^{-7}$ なら相対的には壊れる」という COS 固有の深裾限界は指数写像とは独立に残る)。

**(b) 密度 — 絶対誤差が $e^{-S}$ 倍される(これが「指数の肩」)。**

$$
f_D(D) = f_S(s)\,J(s), \qquad J(s) = \frac{1}{\tilde H_1 - D} = \frac{e^{-s}}{\tilde H_1}.
$$

ヤコビアン $J$ は厳密(誤差を持たない)なので、

$$
\frac{\Delta f_D}{f_D} = \frac{\Delta f_S}{f_S} \ \text{(相対誤差は不変)}, \qquad \Delta f_D = \Delta f_S \cdot \frac{e^{-s}}{\tilde H_1} \ \text{(絶対誤差は } e^{-s} \text{ 倍)}.
$$

$S$ 分布の左端 $A$(= 高ダメージ側)が $D$ 空間の狭いスリバーに**圧縮**され密度が立ち上がるため、$S$ 上でほぼ一様な絶対誤差床 $\varepsilon$ は $D$ 空間で

$$
\Delta f_D \ \lesssim\ \varepsilon \cdot \frac{e^{-A}}{\tilde H_1}, \qquad e^{-A} = \prod_n p_{n,\min}^{-1} = (1 - \beta x_{\max})^{-N}
$$

まで増幅される。これが「構築した $S$ が指数の肩に乗る」ことの定量的な意味で、**増幅は誤差ではなく座標変換として正しい**(密度が本当に高い)——危険なのは密度の*絶対*誤差で精度を判断してしまうこと。相対誤差で見れば不変である。

**増幅率はレジーム依存**で、ミカ設定($\beta = 10^{-6}$、$x_{\max} = 24000$、$p_{\min} = 0.976$、$N=20$)では $e^{-A} = (0.976)^{-20} \approx 1.63$ と穏やか(だから密度も全域で MC と重なる)。一方 $\beta x_{\max} = 0.3$・$N=20$ なら $(0.7)^{-20} \approx 2200$ 倍となり高ダメージ側の密度誤差が深刻化する。**「1 Hit で削る HP 割合 $\beta x$ × Hit 数 $N$」が大きいほど指数の肩が効く**と覚えておくと実用的である。

### 6.3 浮動小数点

数値的に危険なのは増幅される高ダメージ側ではなく**低ダメージ側 $s\approx0$** である。$D/\tilde H_1 \ll 1$ では $1 - D/\tilde H_1 \approx 1$ となり $\ln(\cdot)$ が桁落ちするため、`experiments/product_cos.py` では `np.log1p(-D/H̃_1)` で直接評価している(逆向き $D = \tilde H_1(1-e^S)$ も $S\approx0$ で相殺するので `-expm1(S)` が無難。台の高ダメージ端 $D_{\max} = \tilde H_1(1-e^A)$ の表示にも `expm1` を使用)。

### 6.4 MC 非依存の厳密検証

指数写像のおかげで $D$ のモーメントが閉形式で出るので、MC 標本ノイズゼロの基準が作れる。独立性から $\mathbb E[e^{kS}] = \mathbb E[P^k] = \prod_n \mathbb E[Y_n^k]$(各因子は $U(p,q)$ の積率で閉形式)なので、

$$
\mathbb E[D] = \tilde H_1\Big(1 - \prod_n \mathbb E[Y_n]\Big), \qquad \mathrm{Var}[D] = \tilde H_1^2\Big(\prod_n \mathbb E[Y_n^2] - \big(\textstyle\prod_n \mathbb E[Y_n]\big)^2\Big).
$$

`damage_moments` がこれを返し、COS 再構成の $\int e^s f_S\,ds$(= 指数の肩を含む実効精度)と突き合わせる。ミカ設定では COS の $\mathbb E[D]$ が厳密値と相対誤差 $\sim10^{-13}$ で一致し、MC は標本ノイズで $\approx0.004\%$ ずれる。

## 7. FFT との比較

Qiita の記事は $\Lambda = -S/\beta$ の密度 $q(\Lambda)$ を、各 Hit の $\lambda_n$ 密度を格子上で離散化して FFT で畳み込むことで求めている。FFT 畳み込みは万能だが、(i) 格子の刻みと範囲の取り方で精度が決まり、(ii) Hit 数が増えると畳み込み回数・格子点数が増え、(iii) 深裾は格子の分解能で頭打ちになる。

COS 法は、$\ln Y_n$ の特性関数が**閉形式**で得られることを使い、畳み込みを行わず積 $\prod_n \varphi_{\ln Y_n}$ を直接評価してフーリエ余弦級数で 1 度に反転する。サドルポイントが鞍点での実数評価・求解を要したのに対し、こちらは複素数値の積を取るだけで求解も不要である。項数 $N_t$ は台幅($\sigma$ 単位)に比例し、Hit 数が増えるほど密度は滑らかになるため収束は速やかである。

## 8. 数値結果

`experiments/product_cos.py` で、ミカ設定($R_1=2, R_0=1$、$H = H_1 = 10^6$、20 Hit、基礎ダメージは非会心 $U(8000, 12000)$・会心 $U(16000, 24000)$ を会心率 0.5 で混合)について、HP 依存の漸化式 $H_{n+1} = H_n - (\beta H_n + R_0)x_n$ を直接回した MC(真値)と COS 法を比較した。代表的な健全性チェックは次の通り。

- $F_S(B) = 1.000000$(全質量が台に収まる)
- COS 平均 $\mathrm{E}[S] = -0.30257$ が解析平均 $N\,\mathrm{E}[\ln Y]$ と相対誤差 $10^{-15}$ 程度で一致
- 厳密値 $\mathrm{E}[D] = \tilde H_1(1-\prod_n\mathbb E[Y_n]) = 521{,}727$(MC 非依存)に対し、COS は相対誤差 $\sim10^{-13}$、MC は標本ノイズで $\approx 0.004\%$(§6 を参照)

密度 $f(D)$ は MC ヒストグラムと全域で重なり、上側裾確率 $P(D > x)$ は MC が標本ノイズで荒れる $10^{-5}$ 付近まで COS 法と一致する。それより深い裾では COS 法のほうが滑らかで、`docs/saddlepoint.md` と同じく**準厳密な基準線**として機能する。

## 9. まとめ

- HP 依存ダメージは、シフト HP $\tilde H_n$ の乗法的漸化式により累積ダメージが独立な**積** $P = \prod_n(1 - \beta x_n)$ の単調関数で書ける。
- 対数 $S = \ln P = \sum_n \ln Y_n$ を取れば独立**和**に戻り、`docs/saddlepoint.md` の COS 法がそのまま適用できる。
- $\ln Y_n$ は対数一様(切断指数)分布で、特性関数が閉形式 $\dfrac{q\,e^{iu\ln q} - p\,e^{iu\ln p}}{(q-p)(1+iu)}$ で得られる。
- これは Qiita 記事の FFT 畳み込みを、特性関数の直接反転に置き換えたものに相当し、畳み込み不要・求解不要で準厳密な分布が得られる。

## 参考文献

1. abira, 「ブルアカのHP依存ダメージの確率分布」, Qiita. <https://qiita.com/abira/items/eb93a1b9d6eea5911b99>
2. F. Fang & C. W. Oosterlee, "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions", SIAM Journal on Scientific Computing, 2008.
