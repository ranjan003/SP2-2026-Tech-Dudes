import os
import re

import numpy as np
import pandas as pd

# ==============================================================================
# 🏆 SONA POWERPREDICT - OFFICIAL SAMPLE MODEL TEMPLATE
# ==============================================================================
# Advanced Bayesian + Ridge Regression model for 6-over PowerPlay prediction.
#
# IMPORTANT INSTRUCTIONS:
# 1. Do NOT rename this class. It must be `class MyModel`
# 2. Do NOT rename this file. It must be `mymodelfile.py`
# 3. Your `predict()` method MUST return a pandas DataFrame or List of Dicts
#    containing EXACTLY two columns: ["id", "predicted_score"]
# 4. The test_df uses Numerical Player IDs representing 'striker' and 'bowler'.
#    You must build internal logic in `fit()` to map standard string names to IDs!
# ==============================================================================


class MyModel:

    def __init__(self):
        # --- Player mappings (same as sample template) ---
        self.player_mapping  = {}   # ID (str) → Player_Name
        self.name_to_id      = {}   # Player_Name → ID (str)
        self._id_to_key      = {}   # ID str → normalized lookup key
        self._name_to_key    = {}   # normalized name → normalized lookup key

        # --- Venue medians (same as sample template) ---
        self.venue_median_runs = {}
        self.global_median     = 55.0   # safe fallback for 6-over powerplay

        # --- Entity alias maps ---
        self._team_alias  = {}
        self._venue_alias = {}

        # --- Weighted stat bundles: "all", "core3", "season" ---
        self._bundles = {}

        # --- Ridge regression state ---
        self._reg_weights    = None
        self._reg_feat_means = None
        self._reg_feat_stds  = None
        self._reg_blend      = 0.0
        self._use_regression = False

        # --- Prediction bounds ---
        self._min_score  = 25
        self._max_score  = 110
        self._is_fitted  = False

    # ==========================================================================
    # fit()
    # ==========================================================================

    def fit(self, deliveries_df, players_df=None, matches_df=None):
        """Trains the model using the three provided DataFrames."""

        # --- 1. Map Player Names to IDs ---
        if players_df is not None and not players_df.empty:
            id_col   = self._find_col(players_df.columns, ["ID", "player_id", "unique_id"], required=False)
            name_col = self._find_col(players_df.columns, ["Player_Name", "player_name", "player", "name"], required=False)
            if id_col and name_col:
                self.player_mapping = dict(zip(players_df[id_col].astype(str), players_df[name_col]))
                self.name_to_id     = dict(zip(players_df[name_col], players_df[id_col].astype(str)))
                for row in players_df[[id_col, name_col]].dropna().itertuples(index=False):
                    raw_id = str(row[0]).strip()
                    key    = self._norm(row[1])
                    if raw_id and key:
                        self._id_to_key[raw_id]  = key
                        self._name_to_key[key]   = key
        else:
            print("Warning: players_df is missing.")
            self.name_to_id = {}

        # --- 2. Learn team / venue aliases ---
        self._team_alias  = self._learn_team_aliases(deliveries_df, matches_df)
        self._venue_alias = self._learn_venue_aliases(matches_df)

        # --- 3. Compute venue-specific PP medians ---
        self._compute_venue_medians(deliveries_df, matches_df)

        # --- 4. Calculate Player Aggressiveness (Strike Rates) ---
        self.player_strike_rates = {}
        over_col  = self._find_col(deliveries_df.columns, ["overs", "over"], required=False)
        brun_col  = self._find_col(deliveries_df.columns,
                                   ["batsman_run", "batsman_runs", "runs_off_bat"], required=False)
        strik_col = self._find_col(deliveries_df.columns,
                                   ["striker", "batter", "batsman"], required=False)
        if over_col and brun_col and strik_col:
            overs_raw = pd.to_numeric(deliveries_df[over_col], errors="coerce").fillna(99)
            min_over  = overs_raw[overs_raw > 0].min()
            pp_mask   = overs_raw <= 6 if min_over >= 1 else overs_raw < 6
            pp        = deliveries_df[pp_mask]
            runs      = pp.groupby(strik_col)[brun_col].sum()
            balls     = pp.groupby(strik_col).size()
            for player in balls[balls >= 20].index:
                sr  = (runs[player] / balls[player]) * 100
                pid = self._map_to_id(player)
                self.player_strike_rates[pid] = sr

        # --- 5. Build Bayesian stat bundles ---
        pp_df = self._prepare_pp_deliveries(deliveries_df, matches_df)
        if pp_df.empty:
            print("Warning: no powerplay deliveries found — check over column values")
            return self

        summaries = self._build_summaries(pp_df)
        if not summaries:
            return self

        latest_year = max(r["year"] for r in summaries)
        self._bundles = {
            name: self._build_bundle_stats(summaries, cutoff, latest_year, decay)
            for name, cutoff, decay in [
                ("all",    None,              0.90),
                ("core3",  latest_year - 3,   0.96),
                ("season", latest_year,        1.0),
            ]
        }
        self.global_median = self._bundles["season"]["base_mean"]

        # --- 6. Fit ridge regression ---
        features, targets, weights = [], [], []
        for rec in summaries:
            features.append(self._make_feature_vector(rec, exclude_current=True))
            targets.append(float(rec["target"]))
            weights.append(self._sample_weight(rec["year"], latest_year))

        self._fit_regression(features, targets, weights)
        self._is_fitted = True
        return self

    # ==========================================================================
    # predict()
    # ==========================================================================

    def predict(self, test_df):
        """
        Predicts the total powerplay runs.
        Returns DataFrame with exactly columns ["id", "predicted_score"].
        """
        self._ensure_fitted()

        if test_df is None or len(test_df) == 0:
            return pd.DataFrame(columns=["id", "predicted_score"])

        id_col        = self._find_col(test_df.columns, ["id", "ID"])
        venue_col     = self._find_col(test_df.columns, ["venue", "Venue"],                               required=False)
        innings_col   = self._find_col(test_df.columns, ["innings", "inning"],                            required=False)
        bat_team_col  = self._find_col(test_df.columns, ["batting_team",  "batting team"],                required=False)
        bowl_team_col = self._find_col(test_df.columns, ["bowling_team",  "bowling team"],                required=False)
        batsman_col   = self._find_col(test_df.columns,
                                       ["Batsman's Player Id", "batsman", "batter",
                                        "striker", "batsman_id", "batsman_ids"],                          required=False)
        bowler_col    = self._find_col(test_df.columns,
                                       ["Bowler's Player id (opponent)", "bowler",
                                        "bowler_id", "bowler_ids"],                                       required=False)

        predictions = []
        for _, row in test_df.iterrows():

            # 1. Base on Venue context (same as sample template)
            predicted_score = self.global_median
            venue_raw = str(row.get(venue_col, "") if venue_col else "")
            venue_key = self._resolve_venue(venue_raw)
            if venue_key and venue_key in self.venue_median_runs:
                predicted_score = self.venue_median_runs[venue_key]

            # Map player IDs to names (same as sample template)
            batsman_id   = str(row.get(batsman_col, "") if batsman_col else "")
            batsman_name = self.player_mapping.get(batsman_id, "Unknown")

            # 2. Factor in striker aggressiveness (same as sample template)
            if batsman_id in self.player_strike_rates:
                sr              = self.player_strike_rates[batsman_id]
                sr_factor       = (sr - 120) / 100
                predicted_score += (sr_factor * 5)

            # 3. Override with advanced Bayesian + ridge prediction
            if self._is_fitted and self._bundles:
                record  = self._record_from_row(row, venue_col, innings_col,
                                                bat_team_col, bowl_team_col,
                                                batsman_col, bowler_col)
                fvec    = self._make_feature_vector(record, exclude_current=False)
                predicted_score = self._final_prediction(fvec)

            # 4. Clip to valid range
            final_prediction = int(max(self._min_score, min(self._max_score, predicted_score)))

            predictions.append({
                "id": row[id_col],
                "predicted_score": final_prediction
            })

        return pd.DataFrame(predictions)

    # ==========================================================================
    # Internal — venue median computation
    # ==========================================================================

    def _compute_venue_medians(self, deliveries_df, matches_df):
        """Computes venue-specific PP medians — robust to over indexing."""
        if matches_df is None or matches_df.empty:
            return

        over_col   = self._find_col(deliveries_df.columns, ["overs", "over", "ball"], required=False)
        trun_col   = self._find_col(deliveries_df.columns,
                                    ["total_run", "total_runs", "runs_off_bat"], required=False)
        brun_col   = self._find_col(deliveries_df.columns,
                                    ["batsman_run", "batsman_runs", "runs_off_bat"], required=False)
        extra_col  = self._find_col(deliveries_df.columns,
                                    ["extras_run", "extra_runs", "extras"], required=False)
        inning_col = self._find_col(deliveries_df.columns, ["innings", "inning"], required=False)
        mid_col    = self._find_col(deliveries_df.columns,
                                    ["ID", "matchId", "match_id", "id"], required=False)

        if not (over_col and inning_col and mid_col):
            return

        venue_map = self._build_venue_map(matches_df)
        df = deliveries_df.copy()

        # --- Robust over detection ---
        over_raw   = pd.to_numeric(df[over_col], errors="coerce").fillna(99)
        # Handle cricsheet ball format: "0.1", "1.3" etc → floor to over number
        if df[over_col].astype(str).str.contains(r'^\d+\.\d+$', regex=True).any():
            over_raw = over_raw.apply(np.floor)
        min_over   = over_raw[over_raw >= 0].min()
        # 0-indexed: overs 0–19   → keep < 6
        # 1-indexed: overs 1–20   → keep <= 6
        pp_mask    = (over_raw <= 6) if min_over >= 1 else (over_raw < 6)

        # --- Robust run computation ---
        if trun_col and trun_col != extra_col:
            df["_trun"] = pd.to_numeric(df[trun_col], errors="coerce").fillna(0.0)
        elif brun_col and extra_col:
            df["_trun"] = (pd.to_numeric(df[brun_col], errors="coerce").fillna(0.0) +
                           pd.to_numeric(df[extra_col], errors="coerce").fillna(0.0))
        elif brun_col:
            df["_trun"] = pd.to_numeric(df[brun_col], errors="coerce").fillna(0.0)
        else:
            return

        df["_mid"]   = df[mid_col].astype(str)
        df["_inn"]   = pd.to_numeric(df[inning_col], errors="coerce").fillna(0).astype(int)
        df["_venue"] = df["_mid"].map(venue_map).fillna("")

        pp           = df[pp_mask & df["_inn"].between(1, 2)]
        if pp.empty:
            return

        pp_totals    = pp.groupby(["_mid", "_inn"])["_trun"].sum().reset_index()
        pp_totals["_venue"] = pp_totals["_mid"].map(venue_map).fillna("")

        self.venue_median_runs = (
            pp_totals[pp_totals["_venue"].ne("")]
            .groupby("_venue")["_trun"].median().to_dict()
        )
        self.global_median = float(pp_totals["_trun"].median())

    # ==========================================================================
    # Internal — powerplay delivery preparation (for Bayesian bundles)
    # ==========================================================================

    def _prepare_pp_deliveries(self, deliveries_df, matches_df):
        if deliveries_df is None or deliveries_df.empty:
            return pd.DataFrame()

        # --- Resolve columns (covers all known IPL dataset variants) ---
        mid_col    = self._find_col(deliveries_df.columns,
                                    ["ID", "matchId", "match_id", "id"])
        inn_col    = self._find_col(deliveries_df.columns,
                                    ["innings", "inning"])
        over_col   = self._find_col(deliveries_df.columns,
                                    ["overs", "over", "ball"])
        ball_col   = self._find_col(deliveries_df.columns,
                                    ["ballnumber", "ball_number", "ball"], required=False)
        bteam_col  = self._find_col(deliveries_df.columns,
                                    ["batting_team", "batting team"], required=False)
        wteam_col  = self._find_col(deliveries_df.columns,
                                    ["bowling_team", "bowling team"], required=False)
        bat_col    = self._find_col(deliveries_df.columns,
                                    ["striker", "batter", "batsman"])
        bowl_col   = self._find_col(deliveries_df.columns,
                                    ["bowler"])
        brun_col   = self._find_col(deliveries_df.columns,
                                    ["batsman_run", "batsman_runs", "runs_off_bat"])
        trun_col   = self._find_col(deliveries_df.columns,
                                    ["total_run", "total_runs"], required=False)
        extra_col  = self._find_col(deliveries_df.columns,
                                    ["extras_run", "extra_runs", "extras"], required=False)
        date_col   = self._find_col(deliveries_df.columns,
                                    ["date", "start_date"], required=False)
        dis_col    = self._find_col(deliveries_df.columns,
                                    ["kind", "dismissal_kind", "wicket_type"], required=False)
        dout_col   = self._find_col(deliveries_df.columns,
                                    ["player_out", "player_dismissed"], required=False)

        venue_map  = self._build_venue_map(matches_df)
        df         = deliveries_df.copy()

        # --- Over: auto-detect indexing (0-based vs 1-based vs cricsheet float) ---
        over_raw   = pd.to_numeric(df[over_col], errors="coerce").fillna(99)
        if df[over_col].astype(str).str.match(r'^\d+\.\d+$').any():
            # Cricsheet format: 0.1, 0.2 ... 19.6 → floor = over number (0-indexed)
            over_raw = over_raw.apply(np.floor)
            pp_mask  = over_raw < 6
        else:
            min_over = over_raw[over_raw >= 0].min()
            pp_mask  = (over_raw <= 6) if min_over >= 1 else (over_raw < 6)

        # --- Total runs: compute if not directly available ---
        brun_num = pd.to_numeric(df[brun_col], errors="coerce").fillna(0.0)
        if trun_col:
            trun_num = pd.to_numeric(df[trun_col], errors="coerce").fillna(0.0)
        elif extra_col:
            trun_num = brun_num + pd.to_numeric(df[extra_col], errors="coerce").fillna(0.0)
        else:
            trun_num = brun_num

        # --- Ball number for sorting ---
        if ball_col and ball_col != over_col:
            ball_num = pd.to_numeric(df[ball_col], errors="coerce").fillna(0)
        else:
            # Extract decimal part from cricsheet ball format (e.g. 0.3 → 3)
            ball_num = (over_raw - np.floor(over_raw)).round(1) * 10

        df["pp_mid"]   = df[mid_col].astype(str)
        df["pp_inn"]   = pd.to_numeric(df[inn_col], errors="coerce").fillna(0).astype(int)
        df["pp_over"]  = over_raw.astype(int)
        df["pp_ball"]  = ball_num
        df["pp_brun"]  = brun_num
        df["pp_trun"]  = trun_num
        df["pp_date"]  = pd.to_datetime(df[date_col].fillna("").astype(str),
                                        errors="coerce") if date_col else pd.NaT
        df["pp_year"]  = df["pp_date"].dt.year.fillna(0).astype(int)
        df["pp_bteam"] = df[bteam_col].map(self._resolve_team) if bteam_col else ""
        df["pp_wteam"] = df[wteam_col].map(self._resolve_team) if wteam_col else ""
        df["pp_bkey"]  = df[bat_col].map(self._resolve_player)
        df["pp_wkey"]  = df[bowl_col].map(self._resolve_player)
        df["pp_vkey"]  = df["pp_mid"].map(venue_map).fillna("")
        df["pp_wkt"]   = (
            (df[dis_col].fillna("").astype(str).str.strip().ne("")  if dis_col  else pd.Series(False, index=df.index)) |
            (df[dout_col].fillna("").astype(str).str.strip().ne("") if dout_col else pd.Series(False, index=df.index))
        ).astype(float)

        pp = df[pp_mask & df["pp_inn"].between(1, 2)].copy()
        if pp.empty:
            return pp
        return pp.sort_values(["pp_date", "pp_mid", "pp_inn", "pp_over", "pp_ball"],
                              kind="mergesort").reset_index(drop=True)

    def _build_venue_map(self, matches_df):
        if matches_df is None or matches_df.empty:
            return {}
        id_col    = self._find_col(matches_df.columns,
                                   ["ID", "matchId", "match_id", "id"], required=False)
        venue_col = self._find_col(matches_df.columns,
                                   ["venue", "Venue"], required=False)
        if not id_col or not venue_col:
            return {}
        sub = (matches_df[[id_col, venue_col]]
               .dropna(subset=[id_col])
               .drop_duplicates(subset=[id_col], keep="last"))
        return {str(r[0]): self._resolve_venue(r[1]) for r in sub.itertuples(index=False)}

    # ==========================================================================
    # Internal — inning summary aggregation
    # ==========================================================================

    def _build_summaries(self, pp_df):
        ikeys = ["pp_mid", "pp_inn"]

        meta = (pp_df.groupby(ikeys, sort=False)
                .agg(year=("pp_year", "first"),
                     batting_key=("pp_bteam", "first"),
                     bowling_key=("pp_wteam", "first"),
                     venue_key=("pp_vkey", "first"),
                     target=("pp_trun", "sum"))
                .reset_index())

        batter_orders, batter_stats = {}, {}
        bf = pp_df.loc[pp_df["pp_bkey"].ne("")]
        if not bf.empty:
            batter_orders = (bf.drop_duplicates(ikeys + ["pp_bkey"], keep="first")
                             .groupby(ikeys, sort=False)["pp_bkey"].agg(list).to_dict())
            bagg = (bf.assign(bnd=(bf["pp_brun"] >= 4).astype(float),
                              dot=(bf["pp_trun"] == 0).astype(float),
                              six=(bf["pp_brun"] == 6).astype(float),
                              ball=1)
                    .groupby(ikeys + ["pp_bkey"], sort=False)
                    .agg(runs=("pp_brun", "sum"), balls=("ball", "sum"),
                         bnds=("bnd", "sum"), dots=("dot", "sum"), sixes=("six", "sum"))
                    .reset_index())
            for r in bagg.itertuples(index=False):
                k = (r.pp_mid, r.pp_inn)
                batter_stats.setdefault(k, {})[r.pp_bkey] = [
                    float(r.runs), int(r.balls), float(r.bnds),
                    float(r.dots), float(r.sixes)
                ]

        bowler_orders, bowler_stats = {}, {}
        wf = pp_df.loc[pp_df["pp_wkey"].ne("")]
        if not wf.empty:
            bowler_orders = (wf.drop_duplicates(ikeys + ["pp_wkey"], keep="first")
                             .groupby(ikeys, sort=False)["pp_wkey"].agg(list).to_dict())
            wagg = (wf.assign(dot=(wf["pp_trun"] == 0).astype(float),
                              bnd=(wf["pp_brun"] >= 4).astype(float),
                              ball=1)
                    .groupby(ikeys + ["pp_wkey"], sort=False)
                    .agg(runs=("pp_trun", "sum"), balls=("ball", "sum"),
                         wkts=("pp_wkt", "sum"), dots=("dot", "sum"), bnds=("bnd", "sum"))
                    .reset_index())
            for r in wagg.itertuples(index=False):
                k = (r.pp_mid, r.pp_inn)
                bowler_stats.setdefault(k, {})[r.pp_wkey] = [
                    float(r.runs), int(r.balls), float(r.wkts),
                    float(r.dots), float(r.bnds)
                ]

        summaries = []
        for row in meta.itertuples(index=False):
            key     = (row.pp_mid, row.pp_inn)
            batters = batter_orders.get(key, [])
            bowlers = bowler_orders.get(key, [])
            summaries.append({
                "record_id":    f"{row.pp_mid}_{int(row.pp_inn)}",
                "match_id":     str(row.pp_mid),
                "inning_num":   int(row.pp_inn),
                "year":         int(row.year),
                "batting_key":  row.batting_key,
                "bowling_key":  row.bowling_key,
                "venue_key":    row.venue_key,
                "target":       float(row.target),
                "batter_count": len(batters),
                "bowler_count": len(bowlers),
                "batters":      batters,
                "bowlers":      bowlers,
                "batter_stats": batter_stats.get(key, {}),
                "bowler_stats": bowler_stats.get(key, {}),
            })
        return summaries

    # ==========================================================================
    # Internal — weighted stat bundles
    # ==========================================================================

    def _build_bundle_stats(self, summaries, cutoff_year, latest_year, decay):
        selected = [r for r in summaries if cutoff_year is None or r["year"] >= cutoff_year]

        if selected:
            bw = np.asarray([self._season_wt(r["year"], latest_year, decay) for r in selected])
            base_mean = float(np.average([r["target"] for r in selected], weights=bw))
        else:
            base_mean = self.global_median

        stats = {
            "base_mean": base_mean,
            "record_ids": {r["record_id"] for r in selected},
            "latest_year": latest_year, "decay": decay,
            "groups": {k: {} for k in ["inning", "team", "bowl", "venue",
                                        "venue_inning", "matchup",
                                        "team_venue", "bowl_venue"]},
            "batters": {}, "batters_venue": {},
            "bowlers": {}, "bowlers_venue": {},
            "global_batter": [0.0, 0, 0.0, 0.0, 0.0],
            "global_bowler": [0.0, 0, 0.0, 0.0, 0.0],
        }

        def _add(store, key, val, w):
            if not all(str(k) for k in key):
                return
            slot = store.setdefault(key, [0.0, 0.0])
            slot[0] += float(val) * float(w)
            slot[1] += float(w)

        def _acc(store, key, vals, w):
            if not key:
                return
            slot = store.setdefault(key, [0.0, 0, 0.0, 0.0, 0.0])
            for i, v in enumerate(vals):
                slot[i] += v * float(w)

        for rec in selected:
            tgt = float(rec["target"])
            sw  = self._season_wt(rec["year"], latest_year, decay)
            inn = rec["inning_num"]
            bat = rec["batting_key"]
            bow = rec["bowling_key"]
            vnu = rec["venue_key"]

            _add(stats["groups"]["inning"],       (inn,),          tgt, sw)
            _add(stats["groups"]["team"],         (bat,),          tgt, sw)
            _add(stats["groups"]["bowl"],         (bow,),          tgt, sw)
            _add(stats["groups"]["venue"],        (vnu,),          tgt, sw)
            _add(stats["groups"]["venue_inning"], (vnu, inn),      tgt, sw)
            _add(stats["groups"]["matchup"],      (bat, bow, inn), tgt, sw)
            _add(stats["groups"]["team_venue"],   (bat, vnu, inn), tgt, sw)
            _add(stats["groups"]["bowl_venue"],   (bow, vnu, inn), tgt, sw)

            for pk, vals in rec["batter_stats"].items():
                _acc(stats["batters"],       pk,         vals, sw)
                _acc(stats["batters_venue"], (pk, vnu),  vals, sw)
                for i, v in enumerate(vals):
                    stats["global_batter"][i] += v * sw

            for pk, vals in rec["bowler_stats"].items():
                _acc(stats["bowlers"],       pk,         vals, sw)
                _acc(stats["bowlers_venue"], (pk, vnu),  vals, sw)
                for i, v in enumerate(vals):
                    stats["global_bowler"][i] += v * sw

        return stats

    # ==========================================================================
    # Internal — feature vector construction
    # ==========================================================================

    def _make_feature_vector(self, record, exclude_current):
        s  = self._collect_bundle("season", record, exclude_current)
        c3 = self._collect_bundle("core3",  record, exclude_current)
        at = self._collect_bundle("all",    record, exclude_current)

        bc   = max(2.0, float(record.get("batter_count", 0)))
        wc   = max(2.0, float(record.get("bowler_count", 0)))

        # At PREDICT time: batter_count = lineup depth listed, NOT wickets fallen
        # At TRAIN time:   batter_count = distinct batters = actual wickets fallen
        is_predict = not exclude_current
        wh   = 0.0 if is_predict else max(0.0, bc - 2.0)
        ch   = 0.0 if is_predict else max(0.0, bc - 3.0)
        bctl = max(0.0, 3.0 - wc)
        brot = max(0.0, wc - 3.0)
        bstb = max(0.0, 3.0 - bc)

        # Core estimate — base coefficients normalised to sum ~1.0
        core_est = (
            0.18*s["base"]  + 0.14*c3["base"] + 0.08*at["base"]
            + 0.10*s["inning"]       + 0.12*s["team"]         + 0.06*s["bowl"]
            + 0.09*s["venue"]        + 0.07*s["venue_inning"] + 0.05*s["matchup"]
            + 0.06*c3["team_venue"]  + 0.04*c3["bowl_venue"]  + 0.03*c3["team"]
            + 19.0*s["attack_edge"]
            +  9.0*s["boundary_edge"]
            +  8.0*s["top_attack_edge"]
            + 10.0*(s["g_bowl_wkt"]  - s["bowl_wkt_mean"])
            +  5.0*(s["g_bowl_dot"]  - s["bowl_dot_mean"])
            +  4.0*(s["bat_dot_mean"] - s["g_bat_dot"])
        )
        env_up   = max(0.0, 0.90*(s["base"] - at["base"]) + 0.55*(c3["base"] - at["base"]))
        collapse = (6.0*wh + 4.5*ch + 2.8*bctl
                    + 22.0*max(0.0, s["bowl_wkt_mean"] - s["g_bowl_wkt"])*max(1.0, max(wh, 1.0))
                    + 12.0*max(0.0, s["bowl_dot_mean"] - s["g_bowl_dot"]))
        stab     = (2.5*bstb + 1.4*brot
                    + 8.0*max(0.0, s["attack_edge"])
                    + 4.0*max(0.0, s["top_attack_edge"]))
        heuristic = core_est + env_up + stab - collapse

        return [
            1.0, heuristic,
            s["base"], c3["base"], at["base"],
            s["inning"], s["team"], s["bowl"], s["venue"], s["venue_inning"], s["matchup"],
            s["team_venue"], s["bowl_venue"],
            c3["team"], c3["bowl"], c3["venue"], c3["venue_inning"], c3["matchup"], c3["team_venue"],
            at["team"], at["bowl"], at["venue"], at["venue_inning"],
            s["bat_rpb_mean"], s["bat_rpb_top2"], s["bat_bnd_mean"], s["bat_bnd_top2"],
            s["bat_dot_mean"], s["bat_six_mean"],
            s["bowl_rpb_mean"], s["bowl_rpb_top2"], s["bowl_wkt_mean"], s["bowl_wkt_top2"],
            s["bowl_dot_mean"], s["bowl_bnd_mean"],
            s["attack_edge"], s["top_attack_edge"], s["boundary_edge"],
            c3["bat_rpb_mean"], c3["bat_bnd_mean"], c3["bowl_rpb_mean"], c3["bowl_wkt_mean"],
            c3["attack_edge"],
            at["bat_rpb_mean"], at["bowl_rpb_mean"], at["attack_edge"],
            bc, wc, wh, brot, ch, bctl, bstb,
        ]

    def _collect_bundle(self, name, record, exclude_current):
        stats = self._bundles[name]
        is_ex = exclude_current and record.get("record_id") in stats["record_ids"]
        aug   = dict(record)
        aug["_bly"] = stats["latest_year"]
        aug["_bdc"] = stats["decay"]

        base = stats["base_mean"]
        inn  = record.get("inning_num", 1)
        bat  = record.get("batting_key", "")
        bow  = record.get("bowling_key", "")
        vnu  = record.get("venue_key", "")

        g_bat  = self._global_bat_rates(stats)
        g_bowl = self._global_bowl_rates(stats)
        bat_m  = self._player_metrics(stats, record.get("batters",  []), vnu, "bat",
                                      aug if is_ex else None, g_bat,
                                      stats["latest_year"], stats["decay"])
        bowl_m = self._player_metrics(stats, record.get("bowlers", []), vnu, "bowl",
                                      aug if is_ex else None, g_bowl,
                                      stats["latest_year"], stats["decay"])

        p = {"inning": 22.0, "team": 9.0, "bowl": 9.0, "venue": 9.0,
             "venue_inning": 7.0, "matchup": 6.0, "team_venue": 5.0, "bowl_venue": 5.0}

        def sg(gn, key):
            return self._smooth(stats["groups"][gn], key, aug, is_ex, base, p[gn])

        return {
            "base":           base,
            "inning":         sg("inning",       (inn,)),
            "team":           sg("team",         (bat,)),
            "bowl":           sg("bowl",         (bow,)),
            "venue":          sg("venue",        (vnu,)),
            "venue_inning":   sg("venue_inning", (vnu, inn)),
            "matchup":        sg("matchup",      (bat, bow, inn)),
            "team_venue":     sg("team_venue",   (bat, vnu, inn)),
            "bowl_venue":     sg("bowl_venue",   (bow, vnu, inn)),
            "bat_rpb_mean":   bat_m["mean"][0],  "bat_bnd_mean":   bat_m["mean"][1],
            "bat_dot_mean":   bat_m["mean"][2],  "bat_six_mean":   bat_m["mean"][3],
            "bat_rpb_top2":   bat_m["top2"][0],  "bat_bnd_top2":   bat_m["top2"][1],
            "bowl_rpb_mean":  bowl_m["mean"][0], "bowl_wkt_mean":  bowl_m["mean"][1],
            "bowl_dot_mean":  bowl_m["mean"][2], "bowl_bnd_mean":  bowl_m["mean"][3],
            "bowl_rpb_top2":  bowl_m["top2"][0], "bowl_wkt_top2":  bowl_m["top2"][1],
            "attack_edge":     bat_m["mean"][0] - bowl_m["mean"][0],
            "top_attack_edge": bat_m["top2"][0] - bowl_m["top2"][0],
            "boundary_edge":   bat_m["mean"][1] - bowl_m["mean"][3],
            "g_bat_dot":       g_bat[2],
            "g_bowl_wkt":      g_bowl[1],
            "g_bowl_dot":      g_bowl[2],
        }

    def _smooth(self, store, key, aug, is_ex, base, prior):
        slot = store.get(key)
        if slot is None:
            return base
        total, count = float(slot[0]), float(slot[1])
        if is_ex:
            bw = self._season_wt(aug.get("year", 0), aug.get("_bly", 0), aug.get("_bdc", 1.0))
            total -= float(aug.get("target", 0.0)) * bw
            count -= bw
        if count <= 0:
            return base
        return (total + prior * base) / (count + prior)

    def _player_metrics(self, stats, players, venue_key, kind,
                        current_aug, defaults, latest_year, decay):
        p_store = stats["batters"]       if kind == "bat" else stats["bowlers"]
        v_store = stats["batters_venue"] if kind == "bat" else stats["bowlers_venue"]
        prior   = 18.0
        metrics = []
        for pk in players:
            ov = list(p_store.get(pk, [0.0, 0, 0.0, 0.0, 0.0]))
            vn = list(v_store.get((pk, venue_key), [0.0, 0, 0.0, 0.0, 0.0]))
            if current_aug is not None:
                cs_key = "batter_stats" if kind == "bat" else "bowler_stats"
                if pk in current_aug.get(cs_key, {}):
                    bw = self._season_wt(current_aug.get("year", 0), latest_year, decay)
                    for i, v in enumerate(current_aug[cs_key][pk]):
                        ov[i] -= v * bw
                        vn[i] -= v * bw
            ob = max(0.0, float(ov[1]))
            vb = max(0.0, float(vn[1]))
            om = (self._bayes(ov[0], ob, defaults[0], prior),
                  self._bayes(ov[2], ob, defaults[1], prior),
                  self._bayes(ov[3], ob, defaults[2], prior),
                  self._bayes(ov[4], ob, defaults[3], prior))
            vm = (self._bayes(vn[0], vb, om[0], prior),
                  self._bayes(vn[2], vb, om[1], prior),
                  self._bayes(vn[3], vb, om[2], prior),
                  self._bayes(vn[4], vb, om[3], prior))
            vw = min(0.35, vb / 60.0) if vb > 0 else 0.0
            metrics.append(tuple((1.0 - vw) * om[i] + vw * vm[i] for i in range(4)))
        if not metrics:
            metrics = [defaults]
        arr = np.asarray(metrics, dtype=float)
        return {"mean": np.mean(arr, axis=0), "top2": np.mean(arr[:2], axis=0)}

    # ==========================================================================
    # Internal — ridge regression
    # ==========================================================================

    def _fit_regression(self, features, targets, sample_weights):
        if len(features) < 30:
            self._use_regression = False
            return
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets,  dtype=float)
        w = np.asarray(sample_weights, dtype=float)

        self._reg_feat_means = x[:, 1:].mean(axis=0)
        self._reg_feat_stds  = x[:, 1:].std(axis=0)
        self._reg_feat_stds[self._reg_feat_stds < 1e-8] = 1.0

        xs = x.copy()
        xs[:, 1:] = (xs[:, 1:] - self._reg_feat_means) / self._reg_feat_stds
        sw = np.sqrt(np.maximum(w, 1e-8))[:, None]
        xw = xs * sw
        yw = y * sw.ravel()
        ridge = np.eye(xs.shape[1]) * 3.0
        ridge[0, 0] = 0.0

        try:
            self._reg_weights = np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)
        except np.linalg.LinAlgError:
            self._reg_weights = np.linalg.lstsq(xw.T @ xw + ridge, xw.T @ yw, rcond=None)[0]

        model_preds     = xs @ self._reg_weights
        heuristic_preds = x[:, 1]
        best_blend, best_mae = 0.0, self._wmae(heuristic_preds, y, w)
        for b in (0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0):
            mae = self._wmae(b * model_preds + (1.0 - b) * heuristic_preds, y, w)
            if mae < best_mae:
                best_mae, best_blend = mae, b

        self._reg_blend      = min(0.75, best_blend)
        self._use_regression = self._reg_blend > 0.0

    def _final_prediction(self, fvec):
        heuristic = fvec[1]
        pred = heuristic
        if self._use_regression and self._reg_weights is not None:
            x = np.asarray(fvec, dtype=float)
            x[1:] = (x[1:] - self._reg_feat_means) / self._reg_feat_stds
            model_pred = float(x @ self._reg_weights)
            pred = self._reg_blend * model_pred + (1.0 - self._reg_blend) * heuristic

        # Smart adjustment
        season_base   = float(fvec[2])
        all_time_base = float(fvec[4])
        bowl_rotation = max(0.0, float(fvec[49]))
        bowl_control  = max(0.0, float(fvec[51]))
        bat_stability = max(0.0, float(fvec[52]))
        pred += 0.45 * max(0.0, season_base - all_time_base)
        pred += 2.0  * bowl_rotation
        pred -= 1.5  * bowl_control * bat_stability
        return pred

    # ==========================================================================
    # Internal — entity alias learning
    # ==========================================================================

    def _learn_team_aliases(self, deliveries_df, matches_df):
        meta = {}

        def _ingest(team_series, year_series, venue_series=None):
            teams = team_series.map(self._norm)
            valid = teams.ne("")
            if not valid.any():
                return
            for t, cnt in teams[valid].value_counts(sort=False).items():
                slot = meta.setdefault(t, {"count": 0.0, "last_year": 0, "venues": {}})
                slot["count"] += float(cnt)
                y = int(pd.DataFrame({"t": teams[valid].values, "y": year_series[valid].values})
                        .groupby("t", sort=False)["y"].max().get(t, 0))
                if y > slot["last_year"]:
                    slot["last_year"] = y
            if venue_series is not None:
                v_valid = valid & venue_series.ne("")
                if v_valid.any():
                    vc = (pd.DataFrame({"t": teams[v_valid].values, "v": venue_series[v_valid].values})
                          .value_counts(sort=False))
                    for (t, v), cnt in vc.items():
                        if t and v:
                            meta.setdefault(t, {"count": 0.0, "last_year": 0, "venues": {}})
                            meta[t]["venues"][v] = meta[t]["venues"].get(v, 0.0) + float(cnt)

        if matches_df is not None and not matches_df.empty:
            date_col   = self._find_col(matches_df.columns, ["date", "date1", "Date", "start_date"], required=False)
            season_col = self._find_col(matches_df.columns, ["season", "Season"],                    required=False)
            venue_col  = self._find_col(matches_df.columns, ["venue", "Venue"],                      required=False)
            years      = self._extract_years(matches_df, date_col, season_col)
            venues     = (matches_df[venue_col].map(self._norm_venue)
                          if venue_col else pd.Series([""] * len(matches_df), index=matches_df.index))
            for alias in ["team1", "team2", "toss_winner", "winner"]:
                col = self._find_col(matches_df.columns, [alias], required=False)
                if col:
                    _ingest(matches_df[col], years, venues)

        if deliveries_df is not None and not deliveries_df.empty:
            date_col = self._find_col(deliveries_df.columns, ["date", "start_date"], required=False)
            years    = self._extract_years(deliveries_df, date_col, None)
            for alias in [["batting_team", "batting team"], ["bowling_team", "bowling team"]]:
                col = self._find_col(deliveries_df.columns, alias, required=False)
                if col:
                    _ingest(deliveries_df[col], years)

        labels = sorted(meta)
        if not labels:
            return {}
        tok_sets = {l: set(self._tokens(l)) for l in labels}
        tok_cnt  = {}
        for l in labels:
            for t in tok_sets[l]:
                tok_cnt[t] = tok_cnt.get(t, 0) + 1
        inform    = {l: {t for t in tok_sets[l] if tok_cnt.get(t, 0) <= 2} for l in labels}
        venue_sets = {
            l: {k for k, _ in sorted(meta[l]["venues"].items(), key=lambda x: x[1], reverse=True)[:3]}
            for l in labels
        }

        def _should_merge(a, b):
            return (self._jaccard(inform[a], inform[b]) >= 0.50
                    or (bool(inform[a] & inform[b]) and bool(venue_sets[a] & venue_sets[b])))

        groups = self._union_find(labels, _should_merge)
        alias_map = {}
        for group in groups:
            canonical = max(group, key=lambda l: (meta[l]["last_year"], meta[l]["count"],
                                                   len(tok_sets[l]), len(l)))
            for l in group:
                alias_map[l] = canonical
        return alias_map

    def _learn_venue_aliases(self, matches_df):
        if matches_df is None or matches_df.empty:
            return {}
        venue_col  = self._find_col(matches_df.columns, ["venue", "Venue"],               required=False)
        if not venue_col:
            return {}
        date_col   = self._find_col(matches_df.columns, ["date", "date1", "Date", "start_date"], required=False)
        season_col = self._find_col(matches_df.columns, ["season", "Season"],             required=False)
        city_col   = self._find_col(matches_df.columns, ["city", "City"],                 required=False)
        years      = self._extract_years(matches_df, date_col, season_col)
        cities     = (matches_df[city_col].map(self._norm)
                      if city_col else pd.Series([""] * len(matches_df), index=matches_df.index))
        team_cols  = [self._find_col(matches_df.columns, [a], required=False)
                      for a in ["team1", "team2", "toss_winner", "winner"]]
        team_cols  = [c for c in team_cols if c]

        meta = {}
        for idx, raw in matches_df[venue_col].items():
            key  = self._norm_venue(raw)
            if not key:
                continue
            slot = meta.setdefault(key, {"count": 0.0, "first_year": 9999, "last_year": 0,
                                         "cities": {}, "teams": {}})
            year = int(years.loc[idx]) if idx in years.index else 0
            city = cities.loc[idx]     if idx in cities.index else ""
            slot["count"] += 1.0
            if year > 0:
                slot["first_year"] = min(slot["first_year"], year)
                slot["last_year"]  = max(slot["last_year"],  year)
            if city:
                slot["cities"][city] = slot["cities"].get(city, 0.0) + 1.0
            for col in team_cols:
                t = self._resolve_team(matches_df.loc[idx, col])
                if t:
                    slot["teams"][t] = slot["teams"].get(t, 0.0) + 1.0

        generic = {"stadium", "cricket", "international", "sports", "sport",
                   "association", "academy", "complex", "park", "oval", "ground"}
        labels   = sorted(meta)
        if not labels:
            return {}
        tok_sets  = {l: set(self._tokens(l)) for l in labels}
        tok_cnt   = {}
        for l in labels:
            for t in tok_sets[l]:
                tok_cnt[t] = tok_cnt.get(t, 0) + 1
        inform    = {l: {t for t in tok_sets[l] if t not in generic and tok_cnt.get(t, 0) <= 4}
                     for l in labels}
        city_sets = {l: {k for k, _ in sorted(meta[l]["cities"].items(),
                                               key=lambda x: x[1], reverse=True)[:2]}
                     for l in labels}
        team_sets = {l: {k for k, _ in sorted(meta[l]["teams"].items(),
                                               key=lambda x: x[1], reverse=True)[:3]}
                     for l in labels}

        def _no_overlap(a, b):
            af = meta[a]["first_year"] if meta[a]["first_year"] != 9999 else 0
            bf = meta[b]["first_year"] if meta[b]["first_year"] != 9999 else 0
            if not af or not bf:
                return False
            return meta[a]["last_year"] < bf or meta[b]["last_year"] < af

        def _should_merge(a, b):
            return (self._jaccard(inform[a], inform[b]) >= 0.55
                    or (len(inform[a] & inform[b]) >= 1
                        and bool(city_sets[a] & city_sets[b] or team_sets[a] & team_sets[b]))
                    or (bool(city_sets[a] & city_sets[b]) and bool(team_sets[a] & team_sets[b])
                        and _no_overlap(a, b)))

        groups = self._union_find(labels, _should_merge)
        alias_map = {}
        for group in groups:
            canonical = max(group, key=lambda l: (meta[l]["last_year"], meta[l]["count"],
                                                   len(tok_sets[l]), len(l)))
            for l in group:
                alias_map[l] = canonical
        return alias_map

    # ==========================================================================
    # Internal — small utilities
    # ==========================================================================

    def _record_from_row(self, row, venue_col, innings_col,
                         bat_team_col, bowl_team_col, batsman_col, bowler_col):
        try:
            inn = int(float(row.get(innings_col, 1) if innings_col else 1))
        except Exception:
            inn = 1
        batters = self._parse_players(row.get(batsman_col,  "") if batsman_col  else "")
        bowlers = self._parse_players(row.get(bowler_col,   "") if bowler_col   else "")
        return {
            "record_id":    None,
            "inning_num":   inn, "year": 0,
            "batting_key":  self._resolve_team(row.get(bat_team_col,  "") if bat_team_col  else ""),
            "bowling_key":  self._resolve_team(row.get(bowl_team_col, "") if bowl_team_col else ""),
            "venue_key":    self._resolve_venue(row.get(venue_col,    "") if venue_col     else ""),
            "target":       0.0,
            "batter_count": max(2, len(batters)),
            "bowler_count": max(2, len(bowlers)),
            "batters":      batters, "bowlers": bowlers,
            "batter_stats": {}, "bowler_stats": {},
        }

    def _parse_players(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        text = str(value).strip()
        if not text:
            return []
        seen, result = set(), []
        for token in re.split(r"\s*,\s*", text):
            if not token:
                continue
            key = self._resolve_player(token)
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        return result

    def _resolve_player(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if re.fullmatch(r"\d+\.0+", text):
            text = text.split(".", 1)[0]
        if text.isdigit():
            return self._id_to_key.get(text, text)
        key = self._norm(text)
        return self._name_to_key.get(key, key)

    def _resolve_team(self, value):
        key = self._norm(value)
        return self._team_alias.get(key, key)

    def _resolve_venue(self, value):
        key = self._norm_venue(value)
        return self._venue_alias.get(key, key)

    def _map_to_id(self, val):
        if isinstance(val, str) and not val.isdigit():
            return self.name_to_id.get(val, str(val))
        return str(val)

    def _ensure_fitted(self):
        if self._is_fitted:
            return
        for p in ["/app/training_data/deliveries_updated_ipl_upto_2025.csv",
                  "deliveries_updated_ipl_upto_2025.csv"]:
            if os.path.exists(p):
                print(f"Auto-fitting from {p}")
                self.fit(pd.read_csv(p))
                return
        raise RuntimeError("predict() called before fit(). Call model.fit(deliveries_df) first.")

    def _sample_weight(self, year, latest_year):
        gap = latest_year - year
        for max_gap, weight in [(0, 6.0), (1, 4.0), (2, 2.75), (3, 2.0)]:
            if gap <= max_gap:
                return weight
        return 1.0

    @staticmethod
    def _norm(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip().lower()
        if not text or text == "nan":
            return ""
        text = text.replace("&", " and ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _norm_venue(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        text = text.split("(", 1)[0].strip()
        if "," in text:
            text = text.split(",", 1)[0]
        return MyModel._norm(text)

    @staticmethod
    def _tokens(value):
        return [t for t in MyModel._norm(value).split() if t]

    @staticmethod
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _find_col(columns, aliases, required=True):
        col_map = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in columns}
        for alias in aliases:
            key = re.sub(r"[^a-z0-9]+", "", alias.lower())
            if key in col_map:
                return col_map[key]
        if required:
            raise ValueError(f"Missing required column. Expected one of: {aliases}")
        return None

    @staticmethod
    def _extract_years(df, date_col, season_col):
        if date_col:
            return pd.to_datetime(df[date_col], errors="coerce").dt.year.fillna(0).astype(int)
        if season_col:
            return pd.to_numeric(df[season_col], errors="coerce").fillna(0).astype(int)
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    @staticmethod
    def _union_find(labels, should_merge):
        parent = {l: l for l in labels}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if should_merge(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        groups = {}
        for l in labels:
            groups.setdefault(find(l), []).append(l)
        return list(groups.values())

    @staticmethod
    def _season_wt(year, latest_year, decay):
        if latest_year <= 0:
            return 1.0
        return float(decay) ** float(max(0, int(latest_year) - int(year)))

    @staticmethod
    def _bayes(numerator, denominator, prior_mean, prior_strength):
        denom = max(0.0, float(denominator))
        if denom <= 0:
            return float(prior_mean)
        return (float(numerator) + prior_strength * float(prior_mean)) / (denom + prior_strength)

    @staticmethod
    def _global_bat_rates(stats):
        tr, tb, tbnd, tdot, tsix = stats["global_batter"]
        if tb <= 0:
            return (1.30, 0.22, 0.35, 0.06)
        return (tr / tb, tbnd / tb, tdot / tb, tsix / tb)

    @staticmethod
    def _global_bowl_rates(stats):
        tr, tb, twk, tdot, tbnd = stats["global_bowler"]
        if tb <= 0:
            return (1.35, 0.03, 0.36, 0.22)
        return (tr / tb, twk / tb, tdot / tb, tbnd / tb)

    @staticmethod
    def _wmae(preds, actuals, weights):
        w = np.asarray(weights, dtype=float)
        if w.sum() <= 0:
            return float(np.mean(np.abs(preds - actuals)))
        return float(np.sum(np.abs(preds - actuals) * w) / w.sum())
