# 離散格子分布: Dirichlet 核・厳密な逆 DFT・連続近似の誤差

## 1. 問題設定

`docs/edge.md`・`docs/saddlepoint.md`・`docs/gilpelaez.md` では、1 Hit のダメージを**連続**一様分布 $U(a_j, b_j)$ の混合とみなし、その和 $S_n$ の分布を特性関数の COS 反転で準厳密に計算した。しかし実際のゲームのダメージは整数値であり、本当は**離散**分布である。本稿では「一様分布が連続ではなく離散一様分布(整数格子 $\{a, a+1, \dots, b\}$ 上の一様分布)であるとき、COS 法に何が起きるか」を解析する。

結論を先に述べると、

- 離散一様の特性関数は $\mathrm{sinc}$ ではなく **Dirichlet 核**になり、$u \to \infty$ で減衰せず**周期 $2\pi/g$ で 1 に戻る**(格子幅 $g$)。これは `docs/edge.md` で触れた「離散分布では Cramér 条件が破れる」ことそのもの。
- 密度が存在しない(質量が格子点に集中する)ので、**密度を再構成する版の COS 法は破綻**する(全原子で Gibbs 振動)。正しい対象は確率質量関数(PMF)と階段状 CDF。
- ところが格子分布の CF が周期的であることを逆手にとると、**1 周期 $[-\pi,\pi]/g$ 上の逆 DFT で PMF が厳密に出る**。これは連続 COS の「無限台を有限窓で切る打ち切り誤差」が**そもそも存在しない**、よりクリーンな計算になる。
- 連続近似(これまでの文書のやり方)との差は**エイリアシング誤差**で、Poisson 和公式で定量化できる。格子幅 $g$ が分布の広がり $s_n$ に比べて小さいほど無視でき、ブルアカのダメージスケール($g=1$、$s_n \sim 10^4\text{–}10^5$)では機械精度以下。

記法は既存文書を引き継ぐ。

## 2. 離散一様分布の特性関数 = Dirichlet 核

整数格子 $\{a, a+1, \dots, b\}$ 上の離散一様分布を考える。点数を $N := b - a + 1$、中心を $c := (a+b)/2$ とすると、各点の確率は $1/N$ で、特性関数は等比和(幾何級数)になる。

$$
\varphi_{\mathrm{disc}}(u) = \frac{1}{N}\sum_{k=a}^{b} e^{iuk} = e^{iuc}\,\frac{1}{N}\frac{\sin(Nu/2)}{\sin(u/2)} = e^{iuc}\,D_N(u)
$$

ここで $D_N(u) := \dfrac{\sin(Nu/2)}{N\sin(u/2)}$ は**正規化 Dirichlet 核**である。連続一様 $U(a,b)$ が

$$
\varphi_{\mathrm{cont}}(u) = e^{iuc}\,\frac{\sin(uh)}{uh}, \qquad h = \frac{b-a}{2}
$$

という $\mathrm{sinc}$ 型だったのに対し、離散版は分母の $u/2$ が $\sin(u/2)$ に置き換わった形である。これがすべての違いの源になる。

**周期性と非減衰。** $\mathrm{sinc}$ は $|u|\to\infty$ で $1/u$ で減衰するが、Dirichlet 核は

$$
|D_N(u + 2\pi)| = |D_N(u)|, \qquad |D_N(2\pi m)| = 1 \quad (m \in \mathbb{Z})
$$

と**周期 $2\pi$ で減衰せず**、$u = 2\pi m$ で大きさ 1 のピーク(エイリアス)に戻る。一般に格子幅(増分の最大公約数)が $g$ の格子分布の CF は周期 $2\pi/g$ を持つ。整数格子なら $g=1$、周期 $2\pi$ である。

これは `docs/edge.md` の誤差評価で出てきた **Cramér の条件** $\sup_{|t|\ge\eta_0}|\varphi(t)| \le 1-\delta_0$ が**破れる**ことに他ならない(格子点 $u=2\pi m$ で $|\varphi|=1$)。連続分布で保証された「特性関数が無限遠で減衰する」性質が、離散分布では失われる。

## 3. 密度版 COS が破綻すること

離散分布には密度がない。質量は格子点に集中し、「密度」は Dirac デルタの和

$$
f(x) = \sum_{m} p_m\,\delta(x - m), \qquad p_m = P(S_n = m)
$$

である。`docs/saddlepoint.md` の COS 法は、有限窓 $[a,b]$ 上で**密度を余弦級数に展開**するものだった。

$$
f(x) \approx {\sum_{k=0}^{N_t-1}}' F_k \cos\!\big(u_k(x-a)\big), \qquad F_k = \frac{2}{b-a}\,\mathrm{Re}\!\left[\varphi_{S_n}(u_k)\,e^{-iu_k a}\right]
$$

連続のときは余弦係数 $F_k \sim k^{-n}$ と速く減衰し級数が密度に収束したが、離散だと $\varphi_{S_n}$ が周期的で減衰しないため $F_k$ も**減衰せず**、級数は密度(= デルタの和)に収束しない。各原子の位置で Gibbs 振動が立ち、「密度」は発散する。要するに、**密度を出力に選んだ COS 法は離散分布に対して原理的に使えない**。

正しい出力は階段状の **CDF** か、格子点上の **PMF** である。`docs/gilpelaez.md` で見たとおり、もともと裾確率(CDF)で欲しいものなので、密度を経由しないこちらが本筋である。

## 4. 厳密な離散反転 = 1 周期上の逆 DFT(COS の離散版)

離散一様の和 $S_n = \sum_i X_i$ もまた整数格子上の離散分布で、その台は $\{\sum_i a_i, \dots, \sum_i b_i\}$、CF は独立性から積 $\varphi_{S_n}(u) = \prod_i \varphi_{X_i}(u)$ である(各 $X_i$ が混合なら成分和 $\varphi_{X_i} = \sum_j w_{ij}\varphi_{ij}$ を取る)。積も周期 $2\pi$ を保つ。

ここで周期性を**逆手にとる**。格子分布の反転公式は

$$
p_m = P(S_n = m) = \frac{1}{2\pi}\int_{-\pi}^{\pi} \varphi_{S_n}(u)\,e^{-ium}\,du
$$

で、積分が**有限区間 $[-\pi,\pi]$(ちょうど 1 周期)**で閉じる。連続 COS では「無限台を有限窓 $[a,b]$ で切る打ち切り誤差(キュムラント窓・サポートクリップで管理していたもの)」があったが、**離散ではそれが消える**——周波数積分が真に有限区間だからである。

さらに、被積分関数 $\varphi_{S_n}(u)e^{-ium}$ は $2\pi$ 周期の滑らかな関数なので、$[-\pi,\pi]$ 上の**台形則は指数的に正確**で、しかも等間隔 $M$ 点での台形則は**逆離散フーリエ変換(逆 DFT)そのもの**である。台の点数を $R+1 = \sum_i(b_i-a_i)+1$ とすると、$M \ge R+1$ 点の逆 DFT で全 PMF が**厳密**に(丸め誤差のみで)一度に得られる。FFT を使えば $O(R\log R)$ である。

$$
\boxed{\;p_m = \frac{1}{M}\sum_{l=0}^{M-1} \varphi_{S_n}\!\Big(\frac{2\pi l}{M}\Big)\,e^{-i\,2\pi l m / M}, \qquad M \ge R+1\;}
$$

CDF・裾確率は PMF の累積和 $F(m)=\sum_{j\le m}p_j$ で厳密に出る(Gil-Pelaez を経る必要すらない)。つまり**離散の場合の「COS 法」は逆 DFT であり、連続版より構造的にクリーン**——打ち切り誤差ゼロ、求積誤差ゼロ、原子分離(DP)も不要(原子は格子の自然な構成要素)である。`docs/saddlepoint.md` の miss($h=0$ の退化成分)や DP ハイブリッドは、この格子描像に統一的に吸収される。

唯一のコストは台の点数 $R+1$ で、これは Hit 数 $n$ とダメージ幅に比例する。ダメージが $10^6$ 規模で幅が広いと $R$ が巨大になり、格子をそのまま解像するのは非現実的である。ここで連続近似の出番になる。

## 5. 連続近似の誤差 = エイリアシング(Poisson 和公式)

これまでの文書が離散性を無視して連続一様で計算してきたのは、暗黙に「格子幅 1 はダメージスケールに比べて無視できる」と仮定していたからである。その誤差を定量化する。

離散一様(点 $\{a,\dots,b\}$)を、同じ中心・幅を半セル広げた連続一様 $U(a-\tfrac12,\,b+\tfrac12)$(幅 $N$)と比べると、両者の CF の比は

$$
\frac{\varphi_{\mathrm{disc}}(u)}{\varphi_{\mathrm{cont}}^{[\text{幅}N]}(u)} = \frac{\sin(Nu/2)/(N\sin(u/2))}{\sin(Nu/2)/(Nu/2)} = \frac{u/2}{\sin(u/2)}
$$

と、$N$ にも $c$ にも依らない**純粋に格子(幅 1)由来の因子** $\dfrac{u/2}{\sin(u/2)}$ になる。これは Poisson 和公式が言う**エイリアシング(スペクトルの周期複製)**そのものである。$|u|\ll 1$ では

$$
\frac{u/2}{\sin(u/2)} = 1 + \frac{u^2}{24} + \cdots \approx 1
$$

と 1 に近く、$u$ が $2\pi$ に近づくと $\sin(u/2)\to 0$ で発散して隣のエイリアスを生む。

**COS / 逆反転で実際に評価する周波数の上限 $u_{\max}$ が、エイリアスの立つ $2\pi/g$ より十分手前で、かつそこまでに連続 CF が機械精度まで減衰していれば、離散と連続は機械精度で一致**する。連続一様の積 $\varphi_{\mathrm{cont}}^{S_n} \sim \prod_i (u h_i)^{-1}$ は $u \sim \mathcal{O}(1/h_{\min})$ で深く減衰するので、必要な $u_{\max}$ はダメージ幅 $h$ の逆数スケール、エイリアスは格子幅 $g=1$ の逆数スケール $2\pi$。両者は $h$ 倍も離れている。

ブルアカの数値を入れると、ダメージ幅 $h \sim 10^3\text{–}10^4$、$s_n \sim 10^4\text{–}10^5$ に対し格子幅 $g=1$。連続 CF がエイリアス位置 $u=2\pi$ に達する頃にはとうに $(2\pi h)^{-n}$ 程度——天文学的に小さい——まで落ちているので、**離散性による誤差は機械精度以下**。これが、これまでの文書が安心して連続一様で計算できた理由の厳密な裏付けである。

逆に**離散性が効くのは $g$ が $s_n$ と同程度になるとき**——ダメージ値そのものが小さい整数(数面のサイコロ的なもの)、Hit 数がごく少なく幅も狭い、といった場合である。そのときは前節の逆 DFT で厳密に扱えばよく、台の点数 $R$ も小さいので計算も軽い。**「広がりが大きく格子が細かい領域は連続 COS、格子が粗くて分布が尖る領域は厳密な逆 DFT」**という棲み分けになる。

## 6. モーメント・キュムラントに現れる離散性(Sheppard 補正)

bulk の精度を支配する 2 次の効果も押さえておく。離散一様(点数 $N$)の分散は

$$
\sigma^2_{\mathrm{disc}} = \frac{N^2 - 1}{12}
$$

で、幅 $N$ の連続一様の分散 $N^2/12$ より $1/12$ だけ小さい。一般に格子幅 $g$ なら

$$
\sigma^2_{\mathrm{cont}} = \sigma^2_{\mathrm{disc}} + \frac{g^2}{12} \qquad (\textbf{Sheppard 補正})
$$

である。連続近似で計算する場合、各成分の分散をこの $g^2/12$ 分だけ補正すれば、bulk の合致がさらに良くなる(`docs/edge.md` の中心モーメント $M_2$ に効く)。なお `docs/edge.md` がダメージを $U(l,u)$(幅 $b-a=N-1$)とそのまま連続化しているのは、離散一様 $\{a,\dots,b\}$ を**半セル狭い**連続一様で近似していることに相当し、分散は $(N-1)^2/12$ とさらにずれる。厳密に対応させるなら**両端を $\pm\tfrac12$ 広げた** $U(a-\tfrac12, b+\tfrac12)$ を使うのが正しい連続性補正で、それでも残る $1/12$ が Sheppard 項である。

サドルポイント近似(`docs/saddlepoint.md`)は離散でもそのまま機能する。CGF の素片が $\sinh$ 版の Dirichlet 核

$$
M_{\mathrm{disc}}(t) = e^{tc}\,\frac{\sinh(Nt/2)}{N\sinh(t/2)}, \qquad K_{\mathrm{disc}}(t) = tc + \log\frac{\sinh(Nt/2)}{N\sinh(t/2)}
$$

になるだけで、実軸上では解析的・滑らかなので鞍点方程式は問題なく解ける($u\to 0$ 展開から $\kappa_2 = (N^2-1)/12$ を正しく再現)。ただし裾確率を Lugannani–Rice で出すときは、格子分布向けの**連続性補正版**を使うのが筋で、$\hat u = \hat t\sqrt{K''(\hat t)}$ を

$$
\hat u_{\mathrm{lat}} = \big(1 - e^{-\hat t}\big)\sqrt{K''(\hat t)} \quad\text{(あるいは } 2\sinh(\hat t/2)\sqrt{K''(\hat t)}\text{)}
$$

に置き換える(Daniels 1987 の格子サドルポイント)。これは「格子点の半セルずれ」を $\hat u$ 側で補正するもので、$\hat t \to 0$ で連続版に一致する。

## 7. CDF・裾確率と連続性補正

実用上は、たとえ厳密 PMF を持っていても、連続 COS の CDF と突き合わせたい場面がある。離散 CDF は格子点で段差を持つ階段関数なので、**連続 COS の CDF を半整数点 $m + \tfrac12$ で評価すると離散 CDF $F(m)=P(S_n\le m)$ に最良一致**する(段差の中点で評価する = 連続性補正)。

$$
P(S_n \le m) \;\approx\; F_{\mathrm{COS}}\!\Big(m + \tfrac12\Big)
$$

これは離散分布を正規近似する際の古典的な連続性補正($P(S\le m)\approx \Phi((m+\tfrac12-\mu)/\sigma)$)と同じ思想で、`docs/gilpelaez.md` で見た「線形ランプ = 半ステップ」の段差処理に、さらに格子の半セルを足し込んだものである。深裾で逆 DFT を直接累積する場合は、もちろん補正不要(厳密)である。

## 8. まとめ

- 離散一様の特性関数は $\mathrm{sinc}$ ではなく Dirichlet 核 $e^{iuc}\frac{\sin(Nu/2)}{N\sin(u/2)}$ で、**周期 $2\pi/g$ で減衰しない**(Cramér 条件が破れる)。
- 密度が存在しないので**密度版 COS は全原子で Gibbs を起こして破綻**する。正しい対象は PMF と階段 CDF。
- 格子分布の CF が周期的であることを使えば、**1 周期 $[-\pi,\pi]$ 上の逆 DFT で PMF が厳密に出る**。打ち切り誤差・求積誤差・原子分離がすべて不要な、連続 COS よりクリーンな計算。コストは台の点数 $R+1$。
- 連続近似との差は**エイリアシング誤差**で、$\varphi_{\mathrm{disc}}/\varphi_{\mathrm{cont}} = \frac{u/2}{\sin(u/2)}$。連続 CF がエイリアス位置 $2\pi/g$ までに減衰していれば一致し、**格子幅 $g \ll s_n$ のブルアカのダメージでは機械精度以下**——これまでの連続一様計算が正しかった理由である。
- bulk には Sheppard 補正 $\sigma^2_{\mathrm{cont}} = \sigma^2_{\mathrm{disc}} + g^2/12$、裾には格子版 Lugannani–Rice、CDF 突き合わせには半整数点評価(連続性補正)。離散性が本当に効くのは「ダメージが小さい整数 / ごく少数 Hit で幅も狭い」レジームに限られ、そこは逆 DFT で厳密に扱えばよい。

## 参考文献

1. J. Gil-Pelaez, "Note on the inversion theorem", Biometrika, 1951.
2. F. Fang & C. W. Oosterlee, "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions", SIAM Journal on Scientific Computing, 2008.
3. H. E. Daniels, "Tail probability approximations", International Statistical Review, 1987.(格子サドルポイント・連続性補正)
4. W. F. Sheppard, "On the calculation of the most probable values of frequency-constants…", Proc. London Math. Soc., 1897.(Sheppard 補正)
