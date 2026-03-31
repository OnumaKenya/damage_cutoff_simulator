/**
 * クライアントサイド Monte Carlo シミュレーションエンジン
 * サーバー側 (Python/NumPy) の処理をブラウザ側 JavaScript に移植
 */
(function () {
  "use strict";

  var ns = (window.dash_clientside = window.dash_clientside || {});
  ns.sim = {};

  // =========================================================================
  // 定数
  // =========================================================================
  var N_SAMPLES = 100000;
  var N_CUTOFF_SAMPLES = 100000;

  // 減衰関数セグメント (simulation.py の DAMAGE_FUNC に対応)
  var SEGS = [
    { xMin: 0, xMax: 4000000, a: 1.0, b: 0 },
    { xMin: 4000000, xMax: 6248000, a: 0.8, b: 4000000 - 0.8 * 4000000 },
    { xMin: 6248000, xMax: 8496000, a: 0.65, b: 5798400 - 0.65 * 6248000 },
    { xMin: 8496000, xMax: 10744000, a: 0.5, b: 7259600 - 0.5 * 8496000 },
    {
      xMin: 10744000,
      xMax: 12992000,
      a: 0.4,
      b: 8383600 - 0.4 * 10744000,
    },
    {
      xMin: 12992000,
      xMax: 15240000,
      a: 0.3,
      b: 9282800 - 0.3 * 12992000,
    },
    {
      xMin: 15240000,
      xMax: 17488000,
      a: 0.225,
      b: 9957200 - 0.225 * 15240000,
    },
    {
      xMin: 17488000,
      xMax: 19736000,
      a: 0.15,
      b: 10463000 - 0.15 * 17488000,
    },
    {
      xMin: 19736000,
      xMax: 22000000,
      a: 0.075,
      b: 10800200 - 0.075 * 19736000,
    },
    { xMin: 22000000, xMax: 1e20, a: 0.0, b: 10966999 },
  ];

  // 逆変換テーブル
  var INV = SEGS.map(function (s) {
    return {
      yLo: s.a * s.xMin + s.b,
      yHi: s.a * s.xMax + s.b,
      a: s.a,
      b: s.b,
    };
  });

  // =========================================================================
  // コア数学関数
  // =========================================================================
  function decay(x) {
    // セグメントは xMin 昇順 → x < xMax の最初のセグメントが正解
    for (var i = 0; i < SEGS.length; i++) {
      if (x < SEGS[i].xMax) {
        return SEGS[i].a * x + SEGS[i].b;
      }
    }
    return SEGS[SEGS.length - 1].b; // 上限キャップ
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

  // =========================================================================
  // ヘルパー
  // =========================================================================
  function buildParams(values, ids) {
    var params = {};
    for (var i = 0; i < ids.length; i++) {
      var idx = ids[i].index;
      var param = ids[i].param;
      if (!params[idx]) params[idx] = {};
      params[idx][param] = values[i];
    }
    return params;
  }

  function extractHitParams(
    indices,
    params,
    globalCrit,
    globalEvade,
    damageMode
  ) {
    var hp = {
      critLows: [],
      critHighs: [],
      normalLows: [],
      normalHighs: [],
      critRates: [],
      evadeRates: [],
    };
    for (var ii = 0; ii < indices.length; ii++) {
      var p = params[indices[ii]];
      if (!p) continue;
      var critMin = parseFloat(p.crit_min || 0);
      var critMax = parseFloat(p.crit_max || 0);
      var normalMin = parseFloat(p.normal_min || 0);
      var normalMax = parseFloat(p.normal_max || 0);
      var hits = parseInt(p.hits || 1);
      var cr =
        (p.crit_rate != null
          ? parseFloat(p.crit_rate)
          : parseFloat(globalCrit || 0)) / 100;
      var er =
        (p.evade_rate != null
          ? parseFloat(p.evade_rate)
          : parseFloat(globalEvade || 0)) / 100;

      var rcl, rch, rnl, rnh;
      if (damageMode === "post_decay") {
        rcl = inverseDecay(critMin);
        rch = inverseDecay(critMax);
        rnl = inverseDecay(normalMin);
        rnh = inverseDecay(normalMax);
      } else {
        rcl = critMin;
        rch = critMax;
        rnl = normalMin;
        rnh = normalMax;
      }
      rch = Math.max(rch, rcl);
      rnh = Math.max(rnh, rnl);

      for (var h = 0; h < hits; h++) {
        hp.critLows.push(rcl);
        hp.critHighs.push(rch);
        hp.normalLows.push(rnl);
        hp.normalHighs.push(rnh);
        hp.critRates.push(cr);
        hp.evadeRates.push(er);
      }
    }
    return hp;
  }

  // =========================================================================
  // Monte Carlo シミュレーション
  // =========================================================================
  function simulate(hp, nSamples) {
    var nHits = hp.critLows.length;
    if (nHits === 0) return new Float64Array(nSamples);
    var totals = new Float64Array(nSamples);
    for (var s = 0; s < nSamples; s++) {
      var total = 0;
      for (var h = 0; h < nHits; h++) {
        if (Math.random() < hp.evadeRates[h]) continue;
        var raw;
        if (Math.random() < hp.critRates[h]) {
          raw =
            hp.critLows[h] +
            Math.random() * (hp.critHighs[h] - hp.critLows[h]);
        } else {
          raw =
            hp.normalLows[h] +
            Math.random() * (hp.normalHighs[h] - hp.normalLows[h]);
        }
        total += decay(raw);
      }
      totals[s] = total;
    }
    return totals;
  }

  // =========================================================================
  // ヒストグラム (事前ビニング)
  // =========================================================================
  function computeHistogram(samples, nbins) {
    var min = Infinity,
      max = -Infinity;
    var mean = 0;
    for (var i = 0; i < samples.length; i++) {
      if (samples[i] < min) min = samples[i];
      if (samples[i] > max) max = samples[i];
      mean += samples[i];
    }
    mean /= samples.length;
    if (min === max) max = min + 1;

    var binWidth = (max - min) / nbins;
    var counts = new Array(nbins);
    for (var i = 0; i < nbins; i++) counts[i] = 0;
    for (var i = 0; i < samples.length; i++) {
      var bin = Math.floor((samples[i] - min) / binWidth);
      if (bin >= nbins) bin = nbins - 1;
      counts[bin]++;
    }
    var centers = new Array(nbins);
    for (var i = 0; i < nbins; i++) {
      centers[i] = min + i * binWidth + binWidth / 2;
    }
    return { x: centers, y: counts, binWidth: binWidth, mean: mean };
  }

  // =========================================================================
  // ルックアップテーブル
  // =========================================================================
  function buildLookupTable(sortedSamples, nPoints) {
    nPoints = nPoints || 2000;
    var n = sortedSamples.length;
    if (n === 0) return { values: [], min: 0, max: 0 };
    var step = Math.max(1, Math.floor(n / nPoints));
    var values = [];
    for (var i = 0; i < n; i += step) values.push(sortedSamples[i]);
    return { values: values, min: sortedSamples[0], max: sortedSamples[n - 1] };
  }

  function bisectLeft(arr, val) {
    var lo = 0,
      hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < val) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function exceedanceProb(table, threshold) {
    var values = table.values || [];
    if (!values.length) return 0;
    var idx = bisectLeft(values, threshold);
    var cdf = idx / values.length;
    return Math.round((1 - cdf) * 10000) / 100;
  }

  function valueAtExceedance(table, exceedancePct) {
    var values = table.values || [];
    if (!values.length) return 0;
    var cdf = 1 - exceedancePct / 100;
    var idx = Math.floor(cdf * (values.length - 1));
    idx = Math.max(0, Math.min(idx, values.length - 1));
    return Math.round(values[idx]);
  }

  // =========================================================================
  // 対数スライダー変換
  // =========================================================================
  function logSliderToPct(val) {
    return Math.pow(10, val / 100 - 2);
  }

  function pctToLogSlider(pct) {
    if (pct <= 0) return 0;
    var val = (Math.log10(pct) + 2) * 100;
    return Math.max(0, Math.min(400, val));
  }

  // =========================================================================
  // 数値フォーマット
  // =========================================================================
  function fmt(n) {
    return Math.round(n).toLocaleString();
  }

  // =========================================================================
  // triggered_id パース (pattern-matching callback 用)
  // =========================================================================
  function getTriggeredId() {
    var ctx = window.dash_clientside.callback_context;
    if (!ctx.triggered.length) return null;
    var propId = ctx.triggered[0].prop_id;
    var idStr = propId.substring(0, propId.lastIndexOf("."));
    try {
      return JSON.parse(idStr);
    } catch (e) {
      return idStr;
    }
  }

  function getTriggeredSimpleId() {
    var ctx = window.dash_clientside.callback_context;
    if (!ctx.triggered.length) return null;
    var propId = ctx.triggered[0].prop_id;
    return propId.substring(0, propId.lastIndexOf("."));
  }

  // =========================================================================
  // コールバック: ドラッグ順同期
  // =========================================================================
  ns.sim.syncDragOrder = function (dragOrder, indices) {
    if (!dragOrder) return indices;
    try {
      return JSON.parse(dragOrder);
    } catch (e) {
      return indices;
    }
  };

  // =========================================================================
  // コールバック: 世代カウンタ
  // =========================================================================
  ns.sim.incrementGeneration = function (
    _ci,
    _si,
    _pv,
    _gc,
    _ge,
    _dm,
    currentGen
  ) {
    return (currentGen || 0) + 1;
  };

  // =========================================================================
  // コールバック: 一括適用
  // =========================================================================
  ns.sim.applyGlobal = function (
    nClicks,
    globalCrit,
    globalEvade,
    currentCritValues
  ) {
    var n = currentCritValues.length;
    var crits = [];
    var evades = [];
    for (var i = 0; i < n; i++) {
      crits.push(globalCrit);
      evades.push(globalEvade);
    }
    return [crits, evades];
  };

  // =========================================================================
  // コールバック: シミュレーション実行
  // =========================================================================
  ns.sim.runSimulation = function (
    nClicks,
    values,
    ids,
    sortedIndices,
    cardIndices,
    globalCrit,
    globalEvade,
    targetDamage,
    damageMode
  ) {
    if (!nClicks) throw window.dash_clientside.PreventUpdate;

    var indices =
      sortedIndices && sortedIndices.length ? sortedIndices : cardIndices;
    indices = indices.filter(function (x) {
      return typeof x === "number";
    });

    var params = buildParams(values, ids);
    var hp = extractHitParams(
      indices,
      params,
      globalCrit,
      globalEvade,
      damageMode
    );
    var totalDamage = simulate(hp, N_SAMPLES);

    // ヒストグラム構築
    var hist = computeHistogram(totalDamage, 200);

    var shapes = [
      {
        type: "line",
        x0: hist.mean,
        x1: hist.mean,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { dash: "dash", color: "red" },
      },
    ];
    var annotations = [
      {
        x: hist.mean,
        y: 1,
        yref: "paper",
        text: "期待値: " + fmt(hist.mean),
        showarrow: false,
        yanchor: "bottom",
      },
    ];

    var target = parseFloat(targetDamage || 0);
    var passText = "";
    if (target > 0) {
      shapes.push({
        type: "line",
        x0: target,
        x1: target,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "green" },
      });
      annotations.push({
        x: target,
        y: 1,
        yref: "paper",
        text: "目標: " + fmt(target),
        showarrow: false,
        yanchor: "bottom",
      });
      // 通過率計算
      var passCount = 0;
      for (var i = 0; i < totalDamage.length; i++) {
        if (totalDamage[i] >= target) passCount++;
      }
      var passRate = (passCount / totalDamage.length) * 100;
      passText =
        "目標ダメージ " +
        fmt(target) +
        " の通過確率: " +
        passRate.toFixed(2) +
        "%";
    }

    var figure = {
      data: [
        {
          x: hist.x,
          y: hist.y,
          type: "bar",
          width: hist.binWidth * 0.95,
          name: "ダメージ分布",
        },
      ],
      layout: {
        title: "合計ダメージ分布",
        xaxis: { title: "合計ダメージ" },
        yaxis: { title: "頻度" },
        bargap: 0.05,
        shapes: shapes,
        annotations: annotations,
      },
    };

    return [figure, passText];
  };

  // =========================================================================
  // コールバック: 足切り計算
  // =========================================================================
  ns.sim.computeCutoff = function (
    nClicksList,
    sortedIndices,
    cardIndices,
    values,
    ids,
    globalCrit,
    globalEvade,
    targetDamage,
    damageMode,
    prevDist,
    prevValues,
    generation,
    statusIds
  ) {
    var trigger = getTriggeredId();
    if (
      !trigger ||
      typeof trigger !== "object" ||
      !nClicksList.some(function (n) {
        return n;
      })
    ) {
      throw window.dash_clientside.PreventUpdate;
    }

    var cutoffIndex = trigger.index;
    var key = String(cutoffIndex);
    var order =
      sortedIndices && sortedIndices.length ? sortedIndices : cardIndices;

    var distStore = Object.assign({}, prevDist || {});
    var valuesStore = Object.assign({}, prevValues || {});

    var cached = distStore[key];
    var needRecompute = !cached || cached.generation !== generation;

    var upperTable, lowerTable;
    var statusAction;

    if (needRecompute) {
      var params = buildParams(values, ids);

      // order 内のカットオフ位置で上下に分割
      var marker = "cutoff_" + cutoffIndex;
      var pos = order.indexOf(marker);
      if (pos === -1) pos = order.length;
      var upperIndices = order.slice(0, pos).filter(function (x) {
        return typeof x === "number";
      });
      var lowerIndices = order.slice(pos + 1).filter(function (x) {
        return typeof x === "number";
      });

      // 上側シミュレーション
      var upperHp = extractHitParams(
        upperIndices,
        params,
        globalCrit,
        globalEvade,
        damageMode
      );
      var upperSamples = simulate(upperHp, N_CUTOFF_SAMPLES);
      upperSamples.sort();
      upperTable = buildLookupTable(upperSamples);

      // 下側シミュレーション
      var lowerHp = extractHitParams(
        lowerIndices,
        params,
        globalCrit,
        globalEvade,
        damageMode
      );
      var lowerSamples = simulate(lowerHp, N_CUTOFF_SAMPLES);
      lowerSamples.sort();
      lowerTable = buildLookupTable(lowerSamples);

      distStore[key] = {
        upper_table: upperTable,
        lower_table: lowerTable,
        generation: generation,
      };
      statusAction = "再計算完了";
    } else {
      upperTable = cached.upper_table;
      lowerTable = cached.lower_table;
      statusAction = "キャッシュ利用";
    }

    var target = parseFloat(targetDamage || 0);
    var e2 = 50.0;
    var e1 = valueAtExceedance(upperTable, e2);
    var e3 = Math.max(target - e1, 0);
    var e4 = exceedanceProb(lowerTable, e3);

    valuesStore[key] = {
      e1: Math.round(e1),
      e2: Math.round(e2 * 100) / 100,
      e3: Math.round(e3),
      e4: Math.round(e4 * 100) / 100,
    };

    var upperRange = fmt(upperTable.min) + " ~ " + fmt(upperTable.max);
    var lowerRange = fmt(lowerTable.min) + " ~ " + fmt(lowerTable.max);
    var statusMsg =
      statusAction +
      " | 上側: " +
      upperRange +
      " | 下側: " +
      lowerRange;

    var statusTexts = statusIds.map(function (sid) {
      return sid.index === cutoffIndex
        ? statusMsg
        : window.dash_clientside.no_update;
    });

    return [distStore, valuesStore, statusTexts];
  };

  // =========================================================================
  // コールバック: 足切りスライダー変更
  // =========================================================================
  ns.sim.onSliderChange = function (
    sliderVals,
    currentValues,
    dist,
    targetDamage,
    sliderIds
  ) {
    var trigger = getTriggeredId();
    if (!trigger || typeof trigger !== "object")
      throw window.dash_clientside.PreventUpdate;

    var cutoffKey = String(trigger.index);
    var elem = trigger.elem;

    if (!dist || !dist[cutoffKey]) throw window.dash_clientside.PreventUpdate;

    var idxInList = -1;
    for (var i = 0; i < sliderIds.length; i++) {
      if (
        sliderIds[i].elem === elem &&
        sliderIds[i].index === trigger.index
      ) {
        idxInList = i;
        break;
      }
    }
    if (idxInList === -1) throw window.dash_clientside.PreventUpdate;

    var val = sliderVals[idxInList];
    if (val === null || val === undefined)
      throw window.dash_clientside.PreventUpdate;
    val = parseFloat(val);

    var isPct = elem === "e2" || elem === "e4";
    var pctVal = isPct ? logSliderToPct(val) : val;

    // エコーバック防止
    if (currentValues) {
      var cardValues = currentValues[cutoffKey];
      if (cardValues) {
        var currentStored = parseFloat(
          cardValues[elem] !== undefined ? cardValues[elem] : Infinity
        );
        if (isPct) {
          var currentLog = pctToLogSlider(currentStored);
          if (Math.abs(val - currentLog) < 1.5)
            throw window.dash_clientside.PreventUpdate;
        } else {
          if (Math.abs(val - currentStored) < 0.5)
            throw window.dash_clientside.PreventUpdate;
        }
      }
    }

    var target = parseFloat(targetDamage || 0);
    var upper = dist[cutoffKey].upper_table;
    var lower = dist[cutoffKey].lower_table;

    var e1, e2, e3, e4;
    if (elem === "e1") {
      e1 = pctVal;
      e2 = exceedanceProb(upper, e1);
      e3 = Math.max(target - e1, 0);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e2") {
      e2 = pctVal;
      e1 = valueAtExceedance(upper, e2);
      e3 = Math.max(target - e1, 0);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e3") {
      e3 = pctVal;
      e1 = Math.max(target - e3, 0);
      e2 = exceedanceProb(upper, e1);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e4") {
      e4 = pctVal;
      e3 = valueAtExceedance(lower, e4);
      e1 = Math.max(target - e3, 0);
      e2 = exceedanceProb(upper, e1);
    } else {
      throw window.dash_clientside.PreventUpdate;
    }

    var newValues = Object.assign({}, currentValues || {});
    newValues[cutoffKey] = {
      e1: Math.round(e1),
      e2: Math.round(e2 * 100) / 100,
      e3: Math.round(e3),
      e4: Math.round(e4 * 100) / 100,
    };
    return newValues;
  };

  // =========================================================================
  // コールバック: 足切りスライダー表示更新
  // =========================================================================
  ns.sim.updateCutoffDisplay = function (values, dist, sliderIds) {
    if (!values || !dist) throw window.dash_clientside.PreventUpdate;

    var sVals = [],
      sMins = [],
      sMaxs = [];
    for (var i = 0; i < sliderIds.length; i++) {
      var key = String(sliderIds[i].index);
      var elem = sliderIds[i].elem;
      var cardVals = values[key];
      var cardDist = dist[key];

      if (cardVals) {
        var raw = cardVals[elem] || 0;
        if (elem === "e2" || elem === "e4") {
          sVals.push(pctToLogSlider(parseFloat(raw)));
        } else {
          sVals.push(raw);
        }
      } else {
        sVals.push(window.dash_clientside.no_update);
      }

      if (cardDist) {
        if (elem === "e1") {
          sMins.push(cardDist.upper_table.min || 0);
          sMaxs.push(cardDist.upper_table.max || 10000000);
        } else if (elem === "e3") {
          sMins.push(cardDist.lower_table.min || 0);
          sMaxs.push(cardDist.lower_table.max || 10000000);
        } else {
          sMins.push(window.dash_clientside.no_update);
          sMaxs.push(window.dash_clientside.no_update);
        }
      } else {
        sMins.push(window.dash_clientside.no_update);
        sMaxs.push(window.dash_clientside.no_update);
      }
    }
    return [sVals, sMins, sMaxs];
  };

  // =========================================================================
  // コールバック: % Input 直接入力
  // =========================================================================
  ns.sim.onPctInputChange = function (
    inputVals,
    currentValues,
    dist,
    targetDamage,
    inputIds
  ) {
    var trigger = getTriggeredId();
    if (!trigger || typeof trigger !== "object")
      throw window.dash_clientside.PreventUpdate;

    var cutoffKey = String(trigger.index);
    var elem = trigger.elem;

    if (!dist || !dist[cutoffKey]) throw window.dash_clientside.PreventUpdate;

    var idxInList = -1;
    for (var i = 0; i < inputIds.length; i++) {
      if (
        inputIds[i].elem === elem &&
        inputIds[i].index === trigger.index
      ) {
        idxInList = i;
        break;
      }
    }
    if (idxInList === -1) throw window.dash_clientside.PreventUpdate;

    var val = inputVals[idxInList];
    if (val === null || val === undefined)
      throw window.dash_clientside.PreventUpdate;
    val = Math.max(0.01, Math.min(100, parseFloat(val)));

    // エコーバック防止
    if (currentValues) {
      var cardValues = currentValues[cutoffKey];
      if (cardValues) {
        var currentStored = parseFloat(
          cardValues[elem] !== undefined ? cardValues[elem] : Infinity
        );
        if (Math.abs(val - currentStored) < 0.005)
          throw window.dash_clientside.PreventUpdate;
      }
    }

    var target = parseFloat(targetDamage || 0);
    var upper = dist[cutoffKey].upper_table;
    var lower = dist[cutoffKey].lower_table;

    var e1, e2, e3, e4;
    if (elem === "e2") {
      e2 = val;
      e1 = valueAtExceedance(upper, e2);
      e3 = Math.max(target - e1, 0);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e4") {
      e4 = val;
      e3 = valueAtExceedance(lower, e4);
      e1 = Math.max(target - e3, 0);
      e2 = exceedanceProb(upper, e1);
    } else {
      throw window.dash_clientside.PreventUpdate;
    }

    var newValues = Object.assign({}, currentValues || {});
    newValues[cutoffKey] = {
      e1: Math.round(e1),
      e2: Math.round(e2 * 100) / 100,
      e3: Math.round(e3),
      e4: Math.round(e4 * 100) / 100,
    };
    return newValues;
  };

  // =========================================================================
  // コールバック: Store → % Input 表示同期
  // =========================================================================
  ns.sim.updatePctInputDisplay = function (values, inputIds) {
    if (!values) throw window.dash_clientside.PreventUpdate;
    var out = [];
    for (var i = 0; i < inputIds.length; i++) {
      var key = String(inputIds[i].index);
      var elem = inputIds[i].elem;
      var cardVals = values[key];
      if (cardVals) {
        out.push(Math.round(parseFloat(cardVals[elem] || 0) * 100) / 100);
      } else {
        out.push(window.dash_clientside.no_update);
      }
    }
    return out;
  };

  // =========================================================================
  // コールバック: スライダー → % Input リアルタイム同期
  // =========================================================================
  ns.sim.sliderToPctSync = function (sliderValues, sliderIds, inputIds) {
    var out = [];
    for (var j = 0; j < inputIds.length; j++) {
      var found = false;
      for (var i = 0; i < sliderIds.length; i++) {
        if (
          sliderIds[i].elem === inputIds[j].elem &&
          sliderIds[i].index === inputIds[j].index
        ) {
          var v = sliderValues[i];
          if (v === null || v === undefined) {
            out.push(0.01);
          } else {
            var pct = Math.pow(10, v / 100 - 2);
            out.push(Math.round(pct * 100) / 100);
          }
          found = true;
          break;
        }
      }
      if (!found) out.push(window.dash_clientside.no_update);
    }
    return out;
  };

  // =========================================================================
  // コールバック: マニュアルモーダル開閉
  // =========================================================================
  ns.sim.toggleManualModal = function (openClicks, closeClicks) {
    var id = getTriggeredSimpleId();
    if (id === "open-manual-btn") return { display: "flex" };
    return { display: "none" };
  };
})();
