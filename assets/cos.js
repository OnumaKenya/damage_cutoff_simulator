/**
 * COS 法 + 原子分離 (DP) によるダメージ分布エンジン (クライアントサイド)。
 *
 * app/backend/cos.py の忠実な移植であり、同 .py が正解基準
 * (tests/test_cos.py で検証済み)。理論は docs/saddlepoint.md・docs/discrete.md・
 * docs/product.md を参照。
 *
 * 2 モデル:
 *   - 和モデル (HP非依存): 合計 S_n = Σ X_i を COS 反転。
 *   - 積モデル (HP依存, ミカ型): D = H̃_1(1 - Π Y_n)、S = ln P = Σ ln Y_n を COS 反転。
 *
 * Dash は assets/ の .js をファイル名昇順で読むため、本ファイル (cos.js) は
 * simulation.js より先にロードされ、simulation.js から window.dash_clientside.cos.* を
 * 呼べる。減衰テーブル (SEGS) もここで一元管理し simulation.js から再利用する。
 */
(function () {
  "use strict";

  var ns = (window.dash_clientside = window.dash_clientside || {});
  ns.cos = {};

  // ===========================================================================
  // 減衰テーブル (app/backend/simulation.py の DAMAGE_FUNC が正本)
  // ===========================================================================
  var SEGS = [
    { xMin: 0, xMax: 4000000, a: 1.0, b: 0 },
    { xMin: 4000000, xMax: 6248000, a: 0.8, b: 4000000 - 0.8 * 4000000 },
    { xMin: 6248000, xMax: 8496000, a: 0.65, b: 5798400 - 0.65 * 6248000 },
    { xMin: 8496000, xMax: 10744000, a: 0.5, b: 7259600 - 0.5 * 8496000 },
    { xMin: 10744000, xMax: 12992000, a: 0.4, b: 8383600 - 0.4 * 10744000 },
    { xMin: 12992000, xMax: 15240000, a: 0.3, b: 9282800 - 0.3 * 12992000 },
    { xMin: 15240000, xMax: 17488000, a: 0.225, b: 9957200 - 0.225 * 15240000 },
    { xMin: 17488000, xMax: 19736000, a: 0.15, b: 10463000 - 0.15 * 17488000 },
    { xMin: 19736000, xMax: 22000000, a: 0.075, b: 10800200 - 0.075 * 19736000 },
    { xMin: 22000000, xMax: 1e20, a: 0.0, b: 10966999 },
  ];
  var INV = SEGS.map(function (s) {
    return { yLo: s.a * s.xMin + s.b, yHi: s.a * s.xMax + s.b, a: s.a, b: s.b };
  });

  function decay(x) {
    for (var i = 0; i < SEGS.length; i++) {
      if (x < SEGS[i].xMax) return SEGS[i].a * x + SEGS[i].b;
    }
    return SEGS[SEGS.length - 1].b;
  }
  function decayScalar(x) {
    // Python の _decay_scalar (x_lo <= x < x_hi)。上限超は cap。
    for (var i = 0; i < SEGS.length; i++) {
      if (x >= SEGS[i].xMin && x < SEGS[i].xMax) return SEGS[i].a * x + SEGS[i].b;
    }
    return SEGS[SEGS.length - 1].b;
  }
  function inverseDecay(y) {
    for (var i = 0; i < INV.length; i++) {
      var lo = Math.min(INV[i].yLo, INV[i].yHi);
      var hi = Math.max(INV[i].yLo, INV[i].yHi);
      if (y >= lo && y <= hi) {
        if (INV[i].a === 0) return SEGS[SEGS.length - 1].xMin;
        return (y - INV[i].b) / INV[i].a;
      }
    }
    return y;
  }
  // 減衰後ダメージの上限キャップ (DAMAGE_FUNC 末尾の定数項 = 10966999)。
  var DAMAGE_CAP = SEGS[SEGS.length - 1].b;

  // 安定値 x → 減衰前最小/減衰前最大 の比率 (1 − 1/(1+0.001x) + 0.2)。
  function stabilityMinRatio(x) {
    return 1.0 - 1.0 / (1.0 + 0.001 * x) + 0.2;
  }

  // ダメージ型 1 つの [減衰前下限, 減衰前上限]。最大がキャップに達し安定値が
  // 与えられている場合は、最小から 最大 = 最小 / stabilityMinRatio(x) で逆算する。
  function rawBounds(postMin, postMax, stability, damageMode) {
    if (damageMode !== "post_decay") return [postMin, Math.max(postMax, postMin)];
    var lo = inverseDecay(postMin), hi;
    if (stability != null && postMax >= DAMAGE_CAP) {
      var r = stabilityMinRatio(stability);
      hi = r > 0 ? lo / r : lo;
    } else {
      hi = inverseDecay(postMax);
    }
    return [lo, Math.max(hi, lo)];
  }

  ns.cos.SEGS = SEGS;
  ns.cos.decay = decay;
  ns.cos.inverseDecay = inverseDecay;
  ns.cos.DAMAGE_CAP = DAMAGE_CAP;
  ns.cos.stabilityMinRatio = stabilityMinRatio;
  ns.cos.rawBounds = rawBounds;

  // ===========================================================================
  // 設定
  // ===========================================================================
  var COS_L = 12.0;
  var COS_TERMS_PER_SIGMA = 256;
  var COS_N_MIN = 2048;
  var COS_N_MAX = 1 << 18;
  var ATOM_MERGE_TOL = 1e-6;
  var ATOM_MERGE_TOL_LN = 1e-9;
  var ATOM_MAX = 5000000;

  // ===========================================================================
  // カード → 1Hit 一様混合 (減衰の区分線形分割)
  //   成分は {weight, lo, hi}。退化 (lo==hi) は点質量。
  // ===========================================================================
  function splitUniformThroughDecay(a, b) {
    if (b <= a) {
      var y = decayScalar(a);
      return [{ w: 1.0, lo: y, hi: y }];
    }
    var parts = [];
    var total = b - a;
    for (var i = 0; i < SEGS.length; i++) {
      var lo = Math.max(a, SEGS[i].xMin);
      var hi = Math.min(b, SEGS[i].xMax);
      if (hi <= lo) continue;
      var w = (hi - lo) / total;
      var yLo = SEGS[i].a * lo + SEGS[i].b;
      var yHi = SEGS[i].a * hi + SEGS[i].b;
      if (yHi < yLo) { var t = yLo; yLo = yHi; yHi = t; }
      parts.push({ w: w, lo: yLo, hi: yHi });
    }
    if (!parts.length) return [{ w: 1.0, lo: a, hi: b }];
    return parts;
  }

  /** 1 枚のカード (params) → 1Hit の一様混合 [{weight, lo, hi}]。 */
  function cardToMixture(p, globalCrit, globalEvade, damageMode, globalStab) {
    var critMin = parseFloat(p.crit_min || 0);
    var critMax = parseFloat(p.crit_max || 0);
    var normalMin = parseFloat(p.normal_min || 0);
    var normalMax = parseFloat(p.normal_max || 0);
    var cr = (p.crit_rate != null ? parseFloat(p.crit_rate) : parseFloat(globalCrit || 0)) / 100;
    var er = (p.evade_rate != null ? parseFloat(p.evade_rate) : parseFloat(globalEvade || 0)) / 100;
    // 安定値は全体設定 (サイドバー) から取得する。
    var stab = globalStab != null && globalStab !== "" ? parseFloat(globalStab) : null;

    var cb = rawBounds(critMin, critMax, stab, damageMode);
    var nb = rawBounds(normalMin, normalMax, stab, damageMode);
    var critParts = splitUniformThroughDecay(cb[0], cb[1]);
    var normParts = splitUniformThroughDecay(nb[0], nb[1]);

    var mix = [];
    var i;
    if ((1 - er) * cr > 0) {
      for (i = 0; i < critParts.length; i++)
        mix.push({ weight: (1 - er) * cr * critParts[i].w, lo: critParts[i].lo, hi: critParts[i].hi });
    }
    if ((1 - er) * (1 - cr) > 0) {
      for (i = 0; i < normParts.length; i++)
        mix.push({ weight: (1 - er) * (1 - cr) * normParts[i].w, lo: normParts[i].lo, hi: normParts[i].hi });
    }
    if (er > 0) mix.push({ weight: er, lo: 0.0, hi: 0.0 });

    var totalW = 0;
    for (i = 0; i < mix.length; i++) totalW += mix[i].weight;
    if (totalW > 0) for (i = 0; i < mix.length; i++) mix[i].weight /= totalW;
    return mix;
  }

  /** 指定カード群を [{mix, count}] (カードごとに混合 + Hit数) に展開する。 */
  function buildGroups(indices, params, globalCrit, globalEvade, damageMode, globalStab) {
    var groups = [];
    for (var ii = 0; ii < indices.length; ii++) {
      var p = params[indices[ii]];
      if (!p) continue;
      var hits = parseInt(p.hits || 1);
      if (hits <= 0) continue;
      // 敵の数を Hit 数に掛けて総ヒット数とする (未入力/不正は 1)。
      var enemies = parseInt(p.enemies || 1);
      if (!(enemies >= 1)) enemies = 1;
      groups.push({
        mix: cardToMixture(p, globalCrit, globalEvade, damageMode, globalStab),
        count: hits * enemies,
      });
    }
    return groups;
  }

  // ===========================================================================
  // 複素数配列ユーティリティ ({re: Float64Array, im: Float64Array})
  // ===========================================================================
  function cOnes(n) {
    var re = new Float64Array(n), im = new Float64Array(n);
    for (var k = 0; k < n; k++) re[k] = 1.0;
    return { re: re, im: im };
  }
  function cMulInto(a, b) { // a *= b (elementwise, in place)
    var re = a.re, im = a.im, bre = b.re, bim = b.im, n = re.length;
    for (var k = 0; k < n; k++) {
      var r = re[k] * bre[k] - im[k] * bim[k];
      var i = re[k] * bim[k] + im[k] * bre[k];
      re[k] = r; im[k] = i;
    }
  }
  function cPow(a, count) { // a^count (整数, exponentiation by squaring)
    var n = a.re.length;
    var r = cOnes(n);
    var b = { re: a.re.slice(), im: a.im.slice() };
    while (count > 0) {
      if (count & 1) cMulInto(r, b);
      count >>= 1;
      if (count > 0) cMulInto(b, b);
    }
    return r;
  }

  // ===========================================================================
  // 和モデル: 特性関数
  // ===========================================================================
  function mixtureCF(mix, u) {
    var n = u.length, re = new Float64Array(n), im = new Float64Array(n);
    for (var j = 0; j < mix.length; j++) {
      var w = mix[j].weight, c = 0.5 * (mix[j].lo + mix[j].hi), h = 0.5 * (mix[j].hi - mix[j].lo);
      for (var k = 0; k < n; k++) {
        var uk = u[k];
        var s;
        if (h === 0) s = 1.0;
        else { var uh = uk * h; s = uh === 0 ? 1.0 : Math.sin(uh) / uh; }
        var ang = uk * c;
        re[k] += w * s * Math.cos(ang);
        im[k] += w * s * Math.sin(ang);
      }
    }
    return { re: re, im: im };
  }
  function sumCF(groups, u) {
    var phi = cOnes(u.length);
    for (var g = 0; g < groups.length; g++) {
      cMulInto(phi, cPow(mixtureCF(groups[g].mix, u), groups[g].count));
    }
    return phi;
  }

  // ===========================================================================
  // 和モデル: 純原子部 (点質量成分の DP)
  // ===========================================================================
  function pointComps(mix) {
    var pts = [];
    for (var j = 0; j < mix.length; j++) if (mix[j].hi <= mix[j].lo) pts.push({ w: mix[j].weight, c: mix[j].lo });
    return pts.length ? pts : null;
  }
  function atomCF(groups, u) {
    var phi = cOnes(u.length), n = u.length;
    for (var g = 0; g < groups.length; g++) {
      var pts = pointComps(groups[g].mix);
      if (!pts) { return { re: new Float64Array(n), im: new Float64Array(n) }; }
      var gre = new Float64Array(n), gim = new Float64Array(n);
      for (var pi = 0; pi < pts.length; pi++) {
        for (var k = 0; k < n; k++) {
          var ang = u[k] * pts[pi].c;
          gre[k] += pts[pi].w * Math.cos(ang);
          gim[k] += pts[pi].w * Math.sin(ang);
        }
      }
      cMulInto(phi, cPow({ re: gre, im: gim }, groups[g].count));
    }
    return phi;
  }
  function mergeSorted(vals, probs, tol) {
    // vals は未ソート。ソートして tol 以内をマージ。
    var idx = vals.map(function (_, i) { return i; });
    idx.sort(function (i, j) { return vals[i] - vals[j]; });
    var ov = [], op = [];
    for (var t = 0; t < idx.length; t++) {
      var v = vals[idx[t]], p = probs[idx[t]];
      if (ov.length && v - ov[ov.length - 1] <= tol) op[op.length - 1] += p;
      else { ov.push(v); op.push(p); }
    }
    return { vals: ov, probs: op };
  }
  function atomPartDistribution(groups, tol) {
    var per = [];
    for (var g = 0; g < groups.length; g++) {
      var pts = pointComps(groups[g].mix);
      if (!pts) return null;
      per.push({ pts: pts, count: groups[g].count });
    }
    var vals = [0.0], probs = [1.0];
    for (var i = 0; i < per.length; i++) {
      for (var c = 0; c < per[i].count; c++) {
        var pts2 = per[i].pts;
        if (vals.length * pts2.length > ATOM_MAX) return null;
        var nv = [], np = [];
        for (var a = 0; a < vals.length; a++) {
          for (var b = 0; b < pts2.length; b++) {
            nv.push(vals[a] + pts2[b].c);
            np.push(probs[a] * pts2[b].w);
          }
        }
        var m = mergeSorted(nv, np, tol);
        vals = m.vals; probs = m.probs;
      }
    }
    return { vals: vals, probs: probs };
  }

  // 純原子部の昇順 (値, 累積確率) を作り、階段 CDF を引けるようにする。
  function atomCumulative(atom) {
    var idx = atom.vals.map(function (_, i) { return i; });
    idx.sort(function (i, j) { return atom.vals[i] - atom.vals[j]; });
    var av = [], cum = [0.0];
    for (var t = 0; t < idx.length; t++) {
      av.push(atom.vals[idx[t]]);
      cum.push(cum[cum.length - 1] + atom.probs[idx[t]]);
    }
    return { av: av, cum: cum };
  }
  function atomStepCDF(atomCum, x) {
    // Σ_{av <= x} p。searchsorted side="right"。
    var av = atomCum.av, lo = 0, hi = av.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (av[mid] <= x) lo = mid + 1; else hi = mid; }
    return atomCum.cum[lo];
  }

  // ===========================================================================
  // 和モデル: モーメント・台・COS 区間
  // ===========================================================================
  function hitMoments(mix) {
    var w = 0, mu = 0, j;
    for (j = 0; j < mix.length; j++) mu += mix[j].weight * 0.5 * (mix[j].lo + mix[j].hi);
    var M2 = 0, M4 = 0;
    for (j = 0; j < mix.length; j++) {
      var c = 0.5 * (mix[j].lo + mix[j].hi), h = 0.5 * (mix[j].hi - mix[j].lo);
      var d = c - mu, h2 = h * h;
      M2 += mix[j].weight * (h2 / 3.0 + d * d);
      M4 += mix[j].weight * (h2 * h2 / 5.0 + 2.0 * d * d * h2 + d * d * d * d);
    }
    return { mu: mu, M2: M2, k4: M4 - 3.0 * M2 * M2 };
  }
  function supportBounds(groups) {
    var lo = 0, hi = 0;
    for (var g = 0; g < groups.length; g++) {
      var mlo = Infinity, mhi = -Infinity, mix = groups[g].mix;
      for (var j = 0; j < mix.length; j++) { if (mix[j].lo < mlo) mlo = mix[j].lo; if (mix[j].hi > mhi) mhi = mix[j].hi; }
      lo += groups[g].count * mlo; hi += groups[g].count * mhi;
    }
    return { lo: lo, hi: hi };
  }

  // ===========================================================================
  // COS 級数評価 (Chebyshev/三角漸化式で trig 呼び出しを削減)
  // ===========================================================================
  function cosSeries(FkHalf, a, du, x) {
    // Σ_k FkHalf[k] cos(k·θ),  θ = du·(x−a),  FkHalf[0] は半分済み
    var theta = du * (x - a);
    var ct = Math.cos(theta);
    var sum = FkHalf[0];
    if (FkHalf.length > 1) sum += FkHalf[1] * ct;
    var cm1 = 1.0, ck = ct;
    for (var k = 2; k < FkHalf.length; k++) {
      var cn = 2 * ct * ck - cm1;
      cm1 = ck; ck = cn;
      sum += FkHalf[k] * cn;
    }
    return sum;
  }
  function sinSeriesOverU(Fk, a, du, x) {
    // 0.5·Fk[0]·(x−a) + Σ_{k≥1} Fk[k] sin(k·θ)/(k·du)
    var theta = du * (x - a);
    var st = Math.sin(theta), ct = Math.cos(theta);
    var sum = 0.5 * Fk[0] * (x - a);
    var sm1 = 0.0, sk = st;
    if (Fk.length > 1) sum += Fk[1] * sk / du;
    for (var k = 2; k < Fk.length; k++) {
      var sn = 2 * ct * sk - sm1;
      sm1 = sk; sk = sn;
      sum += Fk[k] * sn / (k * du);
    }
    return sum;
  }

  // ===========================================================================
  // 和モデル: 分布の構築
  // ===========================================================================
  function makeFk(phi, u, a, L) {
    var n = u.length, Fk = new Float64Array(n), two = 2.0 / L;
    for (var k = 0; k < n; k++) {
      var ua = u[k] * a;
      Fk[k] = two * (phi.re[k] * Math.cos(ua) + phi.im[k] * Math.sin(ua));
    }
    return Fk;
  }

  function buildSumDist(groups, dp) {
    if (dp === undefined) dp = true;
    // COS 区間
    var mean = 0, varSum = 0, k4 = 0, g;
    for (g = 0; g < groups.length; g++) {
      var hm = hitMoments(groups[g].mix);
      mean += groups[g].count * hm.mu;
      varSum += groups[g].count * hm.M2;
      k4 += groups[g].count * hm.k4;
    }
    var std = varSum > 0 ? Math.sqrt(varSum) : 0;
    var sb = supportBounds(groups);
    var a, b, widthSigma;
    if (std > 0) {
      var half = COS_L * Math.sqrt(varSum + Math.sqrt(Math.abs(k4)));
      a = Math.max(sb.lo, mean - half);
      b = Math.min(sb.hi, mean + half);
      widthSigma = (b - a) / std;
    } else { a = sb.lo; b = sb.hi; widthSigma = 1.0; }
    var nTerms = Math.ceil(COS_TERMS_PER_SIGMA * widthSigma);
    nTerms = Math.max(COS_N_MIN, Math.min(COS_N_MAX, nTerms));

    var L = b - a, du = Math.PI / L;
    var u = new Float64Array(nTerms);
    for (var k = 0; k < nTerms; k++) u[k] = k * du;

    var phi = sumCF(groups, u);
    var atomCum = null;
    if (dp) {
      var atom = atomPartDistribution(groups, ATOM_MERGE_TOL);
      if (atom && atom.vals.length) {
        var aphi = atomCF(groups, u);
        for (var kk = 0; kk < nTerms; kk++) { phi.re[kk] -= aphi.re[kk]; phi.im[kk] -= aphi.im[kk]; }
        atomCum = atomCumulative(atom);
      }
    }
    var Fk = makeFk(phi, u, a, L);
    var FkHalf = Fk.slice(); FkHalf[0] *= 0.5;

    return {
      kind: "sum", a: a, b: b, du: du, Fk: Fk, FkHalf: FkHalf,
      atomCum: atomCum, supportLo: sb.lo, supportHi: sb.hi,
      mean: mean, std: std,
      cdf: function (x) {
        if (x < this.supportLo) return 0.0;
        if (x >= this.supportHi) return 1.0;
        var v = sinSeriesOverU(this.Fk, this.a, this.du, x);
        if (this.atomCum) v += atomStepCDF(this.atomCum, x);
        return Math.min(1.0, Math.max(0.0, v));
      },
      pdf: function (x) {
        if (x < this.supportLo || x > this.supportHi) return 0.0;
        return Math.max(0.0, cosSeries(this.FkHalf, this.a, this.du, x));
      },
    };
  }

  // ===========================================================================
  // 積モデル (HP依存): Y = 1 - β x、ln Y の特性関数
  // ===========================================================================
  function yMixture(mix, beta) {
    var out = [];
    for (var j = 0; j < mix.length; j++) {
      var y1 = 1.0 - beta * mix[j].lo, y2 = 1.0 - beta * mix[j].hi;
      var lo = Math.min(y1, y2), hi = Math.max(y1, y2);
      if (!(lo > 0.0)) throw new Error("β x ≥ 1 となり Y≤0。HPが負になる設定。");
      out.push({ weight: mix[j].weight, lo: lo, hi: hi });
    }
    return out;
  }
  function logmixCF(ymix, u) {
    var n = u.length, re = new Float64Array(n), im = new Float64Array(n);
    for (var j = 0; j < ymix.length; j++) {
      var w = ymix[j].weight, p = ymix[j].lo, q = ymix[j].hi;
      if (q <= p) {
        var lp0 = Math.log(p);
        for (var k = 0; k < n; k++) { var ang = u[k] * lp0; re[k] += w * Math.cos(ang); im[k] += w * Math.sin(ang); }
      } else {
        var lp = Math.log(p), lq = Math.log(q), dpq = q - p;
        for (var k2 = 0; k2 < n; k2++) {
          var uk = u[k2];
          // num = q e^{i u lq} - p e^{i u lp}
          var nre = q * Math.cos(uk * lq) - p * Math.cos(uk * lp);
          var nim = q * Math.sin(uk * lq) - p * Math.sin(uk * lp);
          // denom = (q-p)(1 + i u)
          var dre = dpq, dim = dpq * uk;
          var den = dre * dre + dim * dim;
          re[k2] += w * (nre * dre + nim * dim) / den;
          im[k2] += w * (nim * dre - nre * dim) / den;
        }
      }
    }
    return { re: re, im: im };
  }
  function cfSHits(ygroups, u) {
    var phi = cOnes(u.length);
    for (var g = 0; g < ygroups.length; g++) cMulInto(phi, cPow(logmixCF(ygroups[g].ymix, u), ygroups[g].count));
    return phi;
  }
  function supportBoundsHits(ygroups) {
    var lo = 0, hi = 0;
    for (var g = 0; g < ygroups.length; g++) {
      var mlo = Infinity, mhi = -Infinity, ym = ygroups[g].ymix;
      for (var j = 0; j < ym.length; j++) { var ll = Math.log(ym[j].lo), hh = Math.log(ym[j].hi); if (ll < mlo) mlo = ll; if (hh > mhi) mhi = hh; }
      lo += ygroups[g].count * mlo; hi += ygroups[g].count * mhi;
    }
    return { lo: lo, hi: hi };
  }
  function logmixMoments(ymix) {
    var mean = 0, ex2 = 0;
    for (var j = 0; j < ymix.length; j++) {
      var w = ymix[j].weight, p = ymix[j].lo, q = ymix[j].hi, m1, e2;
      if (q <= p) { m1 = Math.log(p); e2 = m1 * m1; }
      else {
        var lp = Math.log(p), lq = Math.log(q);
        m1 = (q * lq - p * lp) / (q - p) - 1.0;
        var aq = q * (lq * lq - 2.0 * lq + 2.0), ap = p * (lp * lp - 2.0 * lp + 2.0);
        e2 = (aq - ap) / (q - p);
      }
      mean += w * m1; ex2 += w * e2;
    }
    return { mean: mean, var: ex2 - mean * mean };
  }
  function yRawMoment(ymix, kk) {
    var m = 0;
    for (var j = 0; j < ymix.length; j++) {
      var w = ymix[j].weight, p = ymix[j].lo, q = ymix[j].hi;
      if (q <= p) m += w * Math.pow(p, kk);
      else m += w * (Math.pow(q, kk + 1) - Math.pow(p, kk + 1)) / ((kk + 1) * (q - p));
    }
    return m;
  }
  function damageMomentsHits(ygroups, Htil) {
    var EP = 1, EP2 = 1;
    for (var g = 0; g < ygroups.length; g++) {
      EP *= Math.pow(yRawMoment(ygroups[g].ymix, 1), ygroups[g].count);
      EP2 *= Math.pow(yRawMoment(ygroups[g].ymix, 2), ygroups[g].count);
    }
    return { mean: Htil * (1.0 - EP), var: Htil * Htil * (EP2 - EP * EP) };
  }
  // 積モデルの純原子部 (ln スケール)
  function pointCompsLn(ymix) {
    var pts = [];
    for (var j = 0; j < ymix.length; j++) if (ymix[j].hi <= ymix[j].lo) pts.push({ w: ymix[j].weight, c: Math.log(ymix[j].lo) });
    return pts.length ? pts : null;
  }
  function atomCFHitsLn(ygroups, u) {
    var n = u.length, phi = cOnes(n);
    for (var g = 0; g < ygroups.length; g++) {
      var pts = pointCompsLn(ygroups[g].ymix);
      if (!pts) return { re: new Float64Array(n), im: new Float64Array(n) };
      var gre = new Float64Array(n), gim = new Float64Array(n);
      for (var pi = 0; pi < pts.length; pi++)
        for (var k = 0; k < n; k++) { var ang = u[k] * pts[pi].c; gre[k] += pts[pi].w * Math.cos(ang); gim[k] += pts[pi].w * Math.sin(ang); }
      cMulInto(phi, cPow({ re: gre, im: gim }, ygroups[g].count));
    }
    return phi;
  }
  function atomPartDistributionLn(ygroups) {
    var per = [];
    for (var g = 0; g < ygroups.length; g++) {
      var pts = pointCompsLn(ygroups[g].ymix);
      if (!pts) return null;
      per.push({ pts: pts, count: ygroups[g].count });
    }
    var vals = [0.0], probs = [1.0];
    for (var i = 0; i < per.length; i++) {
      for (var c = 0; c < per[i].count; c++) {
        var pts2 = per[i].pts;
        if (vals.length * pts2.length > ATOM_MAX) return null;
        var nv = [], np = [];
        for (var a = 0; a < vals.length; a++)
          for (var b = 0; b < pts2.length; b++) { nv.push(vals[a] + pts2[b].c); np.push(probs[a] * pts2[b].w); }
        var m = mergeSorted(nv, np, ATOM_MERGE_TOL_LN);
        vals = m.vals; probs = m.probs;
      }
    }
    return { vals: vals, probs: probs };
  }

  function buildProductDist(ygroups, hp, dp) {
    if (dp === undefined) dp = true;
    var sb = supportBoundsHits(ygroups);
    var a = sb.lo, b = sb.hi;
    var meanS = 0, varS = 0, g;
    for (g = 0; g < ygroups.length; g++) {
      var lm = logmixMoments(ygroups[g].ymix);
      meanS += ygroups[g].count * lm.mean; varS += ygroups[g].count * lm.var;
    }
    var stdS = varS > 0 ? Math.sqrt(varS) : (b - a);
    var widthSigma = stdS > 0 ? (b - a) / stdS : 1.0;
    var nTerms = Math.max(1024, Math.min(COS_N_MAX, Math.ceil(COS_TERMS_PER_SIGMA * widthSigma)));

    var L = b - a, du = Math.PI / L;
    var u = new Float64Array(nTerms);
    for (var k = 0; k < nTerms; k++) u[k] = k * du;

    var phi = cfSHits(ygroups, u);
    var atomCum = null;
    if (dp) {
      var atom = atomPartDistributionLn(ygroups);
      if (atom && atom.vals.length) {
        var aphi = atomCFHitsLn(ygroups, u);
        for (var kk = 0; kk < nTerms; kk++) { phi.re[kk] -= aphi.re[kk]; phi.im[kk] -= aphi.im[kk]; }
        atomCum = atomCumulative(atom);
      }
    }
    var Fk = makeFk(phi, u, a, L);
    var FkHalf = Fk.slice(); FkHalf[0] *= 0.5;
    var Htil = hp.Htil;
    var dMax = -Htil * Math.expm1(a);
    var dm = damageMomentsHits(ygroups, Htil);

    return {
      kind: "product", a: a, b: b, du: du, Fk: Fk, FkHalf: FkHalf,
      atomCum: atomCum, Htil: Htil, dMax: dMax,
      supportLo: 0.0, supportHi: dMax,
      mean: dm.mean, std: Math.sqrt(Math.max(dm.var, 0)),
      _cdfS: function (s) {
        var v = sinSeriesOverU(this.Fk, this.a, this.du, s);
        if (this.atomCum) v += atomStepCDF(this.atomCum, s);
        return Math.min(1.0, Math.max(0.0, v));
      },
      cdf: function (d) {
        if (d < 0) return 0.0;
        var arg = 1.0 - d / this.Htil;
        if (!(arg > 0.0)) return 1.0;           // d >= Htil
        var s = Math.log1p(-d / this.Htil);
        if (s < this.a) return 1.0;             // 高ダメージ端を超過
        if (s > this.b) return 0.0;
        return Math.min(1.0, Math.max(0.0, 1.0 - this._cdfS(s)));
      },
      pdf: function (d) {
        if (d < 0) return 0.0;
        var arg = 1.0 - d / this.Htil;
        if (!(arg > 0.0)) return 0.0;
        var s = Math.log1p(-d / this.Htil);
        if (s < this.a || s > this.b) return 0.0;
        return Math.max(0.0, cosSeries(this.FkHalf, this.a, this.du, s) / Math.abs(this.Htil - d));
      },
    };
  }

  // ===========================================================================
  // 公開 API
  // ===========================================================================
  function hpParams(hp) {
    var H = parseFloat(hp.H), H1 = parseFloat(hp.H1), R0 = parseFloat(hp.R0), R1 = parseFloat(hp.R1);
    var dR = R1 - R0, beta = dR / H;
    return { H: H, H1: H1, R0: R0, R1: R1, dR: dR, beta: beta, Htil: H1 + R0 / beta };
  }

  /**
   * カード設定から分布オブジェクトを構築する。
   * opts: {indices, params, globalCrit, globalEvade, globalStability, damageMode, hpMode, hp}
   *   hpMode === 'on' なら積モデル (hp = {H,H1,R0,R1})、それ以外は和モデル。
   * 返り値は {kind, mean, std, supportLo, supportHi, dMax?, cdf(x), pdf(x)}。
   */
  ns.cos.buildDist = function (opts) {
    var groups = buildGroups(opts.indices, opts.params, opts.globalCrit, opts.globalEvade, opts.damageMode, opts.globalStability);
    if (!groups.length) return null;
    if (opts.hpMode === "on") {
      var hp = hpParams(opts.hp);
      var ygroups = groups.map(function (gr) { return { ymix: yMixture(gr.mix, hp.beta), count: gr.count }; });
      return buildProductDist(ygroups, hp, true);
    }
    return buildSumDist(groups, true);
  };

  /** 分布を細グリッド上の {x, pdf, cdf} に評価する (図用)。 */
  ns.cos.distribution = function (opts, nGrid) {
    var dist = ns.cos.buildDist(opts);
    if (!dist) return null;
    nGrid = nGrid || 600;
    var lo = dist.supportLo, hi = dist.supportHi;
    // 和モデルは描画範囲を [mean±k·σ] ∩ サポートに絞る (台が広いと粗くなるため)
    if (dist.kind === "sum" && dist.std > 0) {
      lo = Math.max(lo, dist.mean - 8 * dist.std);
      hi = Math.min(hi, dist.mean + 8 * dist.std);
    }
    var x = new Float64Array(nGrid), pdf = new Float64Array(nGrid), cdf = new Float64Array(nGrid);
    var step = (hi - lo) / (nGrid - 1);
    for (var i = 0; i < nGrid; i++) {
      var xi = lo + i * step;
      x[i] = xi; pdf[i] = dist.pdf(xi); cdf[i] = dist.cdf(xi);
    }
    return {
      x: Array.prototype.slice.call(x),
      pdf: Array.prototype.slice.call(pdf),
      cdf: Array.prototype.slice.call(cdf),
      mean: dist.mean, std: dist.std,
      supportLo: dist.supportLo, supportHi: dist.supportHi,
    };
  };

  /**
   * 足切り用の CDF テーブル {grid, cdf, min, max} (JSON シリアライズ可)。
   * grid 上の単調 CDF を保持し、exceedanceProb / valueAtExceedance で補間する。
   */
  ns.cos.cdfTable = function (opts, nGrid) {
    var dist = ns.cos.buildDist(opts);
    if (!dist) return { grid: [], cdf: [], min: 0, max: 0 };
    nGrid = nGrid || 2000;
    var lo = dist.supportLo, hi = dist.supportHi;
    var grid = new Array(nGrid), cdf = new Array(nGrid);
    var step = (hi - lo) / (nGrid - 1);
    for (var i = 0; i < nGrid; i++) {
      var xi = lo + i * step;
      grid[i] = xi; cdf[i] = dist.cdf(xi);
    }
    return { grid: grid, cdf: cdf, min: lo, max: hi, mean: dist.mean };
  };

  // CDF テーブル補間 (grid 昇順, cdf 単調非減少)
  function interpCdf(table, x) {
    var grid = table.grid, cdf = table.cdf, n = grid.length;
    if (!n) return 0;
    if (x <= grid[0]) return cdf[0];
    if (x >= grid[n - 1]) return cdf[n - 1];
    var lo = 0, hi = n - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (grid[mid] <= x) lo = mid; else hi = mid; }
    var t = (x - grid[lo]) / (grid[hi] - grid[lo]);
    return cdf[lo] + t * (cdf[hi] - cdf[lo]);
  }
  // 超過確率 P(X >= x) を % で返す
  ns.cos.exceedanceProb = function (table, threshold) {
    if (!table || !table.grid || !table.grid.length) return 0;
    var c = interpCdf(table, threshold);
    return Math.round((1 - c) * 10000) / 100;
  };
  // 指定超過確率に対応するダメージ (単調 CDF を逆引き)
  ns.cos.valueAtExceedance = function (table, exceedancePct) {
    var grid = table.grid, cdf = table.cdf, n = grid ? grid.length : 0;
    if (!n) return 0;
    var targetCdf = 1 - exceedancePct / 100;
    if (targetCdf <= cdf[0]) return Math.round(grid[0]);
    if (targetCdf >= cdf[n - 1]) return Math.round(grid[n - 1]);
    var lo = 0, hi = n - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (cdf[mid] <= targetCdf) lo = mid; else hi = mid; }
    var denom = cdf[hi] - cdf[lo];
    var t = denom > 0 ? (targetCdf - cdf[lo]) / denom : 0;
    return Math.round(grid[lo] + t * (grid[hi] - grid[lo]));
  };

  // ソート済み MC サンプル → {grid, cdf, min, max} (MC 経路を COS と同じ表に揃える)
  ns.cos.tableFromSamples = function (sortedSamples, nGrid) {
    var n = sortedSamples.length;
    if (!n) return { grid: [], cdf: [], min: 0, max: 0 };
    nGrid = nGrid || 2000;
    var lo = sortedSamples[0], hi = sortedSamples[n - 1];
    var grid = new Array(nGrid), cdf = new Array(nGrid);
    var step = (hi - lo) / (nGrid - 1);
    for (var i = 0; i < nGrid; i++) {
      var xi = lo + i * step;
      // 経験 CDF: searchsorted(side='right')/n
      var a = 0, b = n;
      while (a < b) { var mid = (a + b) >> 1; if (sortedSamples[mid] <= xi) a = mid + 1; else b = mid; }
      grid[i] = xi; cdf[i] = a / n;
    }
    return { grid: grid, cdf: cdf, min: lo, max: hi };
  };
})();
