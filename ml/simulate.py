"""
Monte-Carlo simulation of the 2026 FIFA World Cup.

Engine
------
For every match the combined model (Dixon-Coles + gradient-boosted ensemble)
supplies the outcome distribution (p_home, p_draw, p_away). Each simulation:

  GROUP STAGE  (12 groups × 6 = 72 matches)
    • sample an outcome from the combined model
    • sample a scoreline from the Dixon-Coles scoreline matrix, restricted to
      the cells consistent with the sampled outcome
    • award 3 / 1 / 0 points and rank each group with FIFA tiebreakers
      (pts → GD → GF → head-to-head pts → H2H GD → [fair-play skipped] → random)
    • the 12 winners + 12 runners-up + 8 best third-placed teams advance (32)

  KNOCKOUT  (Round of 32 → R16 → QF → SF → Final)
    • single-elimination on a fixed 32-slot bracket
    • a draw after 90' goes to extra time (draw mass halved) then, if still
      level, penalties (50/50 simplification)

Run
---
    python3 ml/simulate.py                # 50,000 sims (default)
    WC_SIMS=500 python3 ml/simulate.py    # quick smoke test

Outputs (data/processed/)
-------------------------
    group_standings.json        per-group expected pts/GD + advancement %
    match_predictions.json      the 72 group matches, model predictions
    knockout_probabilities.json per-team round-reached distribution
    tournament_winner_odds.json top winners with implied decimal odds
    bracket.json                most-likely occupant of every bracket slot
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_model as cm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROUPS_PATH   = Path(os.environ.get("WC_GROUPS", "data/fixtures/groups.json"))
MARKET_PATH   = Path("data/fixtures/wc2026_winner_odds.json")
PROCESSED     = Path("data/processed")
N_SIMS        = int(os.environ.get("WC_SIMS", "50000"))
SEED          = 42
MARKET_BLEND  = 0.25      # weight on the bookmaker prior in the final winner odds
                          # (0.75 on the data-driven simulation)

COMBOS = list(combinations(range(4), 2))   # 6 group matches per group
ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Winner"]
# (round name, number of matches played in that round)
KO_ROUNDS = [("R16", 16), ("QF", 8), ("SF", 4), ("Final", 2), ("Champion", 1)]
LABELS = ["home_win", "draw", "away_win"]

# Knockout bracket
# ----------------
# WC2026 advances 12 group winners + 12 runners-up + 8 best third-placed teams
# (32 → Round of 32). FIFA's official mapping of group-finish position to bracket
# slot has NOT been published for 2026, so any fixed assignment is arbitrary and
# — worse — systematically biased: it can force several strong groups' winners
# into the same half. To avoid that, each simulation draws a FRESH RANDOM bracket
# over the 32 qualifiers (with no same-group Round-of-32 meeting, per FIFA rules).
# Averaged over many sims this marginalises tournament odds over every possible
# bracket and removes half-seeding bias. Replace draw_bracket() with the official
# slot table once it is released if an exact single bracket is required.


# ── Setup ─────────────────────────────────────────────────────────────────────

def load_groups() -> dict[str, list[str]]:
    data = json.loads(GROUPS_PATH.read_text())["groups"]
    groups = {k: list(v) for k, v in data.items()}

    problems = []
    if len(groups) != 12:
        problems.append(f"expected 12 groups, found {len(groups)}")
    for g, teams in groups.items():
        if len(teams) != 4:
            problems.append(f"group {g} has {len(teams)} teams (need 4)")
        for t in teams:
            if t.upper() == "TBD":
                problems.append(f"group {g} still has a TBD placeholder")
    if problems:
        for p in problems:
            log.error("  groups.json: %s", p)
        log.error("Fill in data/fixtures/groups.json with the confirmed 12×4 draw, then re-run.")
        sys.exit(1)
    return groups


def validate_team_names(groups: dict[str, list[str]], dc) -> None:
    missing = [t for ts in groups.values() for t in ts if t not in dc.alpha]
    if missing:
        log.error("Teams not found in Dixon-Coles model: %s", missing)
        log.error("Fix the names in groups.json (must match the model exactly).")
        sys.exit(1)
    log.info("All 48 team names matched the model. ✓")


def load_market_prior() -> tuple[dict[str, float], dict]:
    """Normalised bookmaker outright-winner probabilities (market prior), if present."""
    if not MARKET_PATH.exists():
        log.warning("No market prior at %s — winner odds will be pure simulation.", MARKET_PATH)
        return {}, {}
    data = json.loads(MARKET_PATH.read_text())
    market = {o["team"]: float(o["market_prob"]) for o in data["odds"]}
    meta = {k: data[k] for k in ("source", "source_url", "collected_date") if k in data}
    return market, meta


def conditional_scoreline_cells(mat: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Split a Dixon-Coles scoreline matrix into per-outcome (cells, probs)."""
    n = mat.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    out = {}
    for cat, mask in ((0, ii > jj), (1, ii == jj), (2, ii < jj)):
        cells = np.stack([ii[mask], jj[mask]], axis=1).astype(np.int16)
        probs = mat[mask].astype(float)
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(len(probs), 1 / len(probs))
        out[cat] = (cells, probs)
    return out


# ── Group-match precomputation & vectorised sampling ──────────────────────────

class GroupMatch:
    __slots__ = ("g", "a", "b", "probs", "cond", "out", "gh", "ga")

    def __init__(self, g, a, b, probs, cond):
        self.g, self.a, self.b = g, a, b
        self.probs = probs               # (p_home, p_draw, p_away)
        self.cond = cond                 # {0/1/2: (cells, probs)}
        self.out = self.gh = self.ga = None

    def sample(self, rng: np.random.Generator, n: int) -> None:
        self.out = rng.choice(3, size=n, p=self.probs).astype(np.int8)
        self.gh = np.zeros(n, np.int8)
        self.ga = np.zeros(n, np.int8)
        for cat, (cells, cprobs) in self.cond.items():
            mask = self.out == cat
            k = int(mask.sum())
            if not k:
                continue
            sel = rng.choice(len(cells), size=k, p=cprobs)
            self.gh[mask] = cells[sel, 0]
            self.ga[mask] = cells[sel, 1]


def build_group_matches(groups: dict[str, list[str]], dc) -> dict[str, list[GroupMatch]]:
    gm: dict[str, list[GroupMatch]] = {}
    for g, teams in groups.items():
        matches = []
        for a, b in COMBOS:
            r = cm.predict_match(teams[a], teams[b], neutral=True)
            probs = np.array([r["home_win_prob"], r["draw_prob"], r["away_win_prob"]])
            probs = probs / probs.sum()
            mat = dc.predict_scoreline_probs(teams[a], teams[b], neutral=True)
            matches.append(GroupMatch(g, a, b, probs, conditional_scoreline_cells(mat)))
        gm[g] = matches
    return gm


# ── Group standings (vectorised aggregates + per-sim ranking) ─────────────────

def group_aggregates(gm: dict[str, list[GroupMatch]], n: int):
    """Return per-group (12, 4, n) arrays of points, GF, GA."""
    letters = sorted(gm)
    P  = np.zeros((12, 4, n), np.int16)
    GF = np.zeros((12, 4, n), np.int16)
    GA = np.zeros((12, 4, n), np.int16)
    for gi, g in enumerate(letters):
        for m in gm[g]:
            home_pts = np.where(m.out == 0, 3, np.where(m.out == 1, 1, 0)).astype(np.int16)
            away_pts = np.where(m.out == 2, 3, np.where(m.out == 1, 1, 0)).astype(np.int16)
            P[gi, m.a] += home_pts
            P[gi, m.b] += away_pts
            GF[gi, m.a] += m.gh; GA[gi, m.a] += m.ga
            GF[gi, m.b] += m.ga; GA[gi, m.b] += m.gh
    return letters, P, GF, GA


def rank_group(p, gd, gf, matches, s, rng) -> list[int]:
    """Rank the 4 local team indices for sim s applying FIFA tiebreakers."""
    order = sorted(range(4), key=lambda t: (p[t], gd[t], gf[t]), reverse=True)
    # Resolve clusters tied on (pts, GD, GF).
    i = 0
    while i < 4:
        j = i
        while j + 1 < 4 and (p[order[j + 1]], gd[order[j + 1]], gf[order[j + 1]]) \
                == (p[order[i]], gd[order[i]], gf[order[i]]):
            j += 1
        if j > i:
            order[i:j + 1] = _break_tie(order[i:j + 1], matches, s, rng)
        i = j + 1
    return order


def _break_tie(cluster: list[int], matches, s, rng) -> list[int]:
    """Head-to-head pts → H2H GD → random, among the tied teams only."""
    cset = set(cluster)
    h2h_pts = dict.fromkeys(cluster, 0)
    h2h_gd  = dict.fromkeys(cluster, 0)
    for m in matches:
        if m.a in cset and m.b in cset:
            gh, ga = int(m.gh[s]), int(m.ga[s])
            if gh > ga:
                h2h_pts[m.a] += 3
            elif gh < ga:
                h2h_pts[m.b] += 3
            else:
                h2h_pts[m.a] += 1; h2h_pts[m.b] += 1
            h2h_gd[m.a] += gh - ga
            h2h_gd[m.b] += ga - gh
    return sorted(cluster, key=lambda t: (h2h_pts[t], h2h_gd[t], rng.random()), reverse=True)


# ── Knockout ──────────────────────────────────────────────────────────────────

_KO_CACHE: dict[tuple[str, str], np.ndarray] = {}


def ko_probs(home: str, away: str) -> np.ndarray:
    key = (home, away)
    p = _KO_CACHE.get(key)
    if p is None:
        r = cm.predict_match(home, away, neutral=True)
        p = np.array([r["home_win_prob"], r["draw_prob"], r["away_win_prob"]])
        p = p / p.sum()
        _KO_CACHE[key] = p
    return p


def prewarm_ko_cache(teams: list[str]) -> None:
    """Predict every ordered pair once so the per-sim loop never goes cold."""
    for h in teams:
        for a in teams:
            if h != a:
                ko_probs(h, a)


def ko_winner(home: str, away: str, rng: np.random.Generator) -> str:
    p = ko_probs(home, away)               # (p_home, p_draw, p_away)
    u = rng.random()
    if u < p[0]:
        return home
    if u >= p[0] + p[1]:
        return away
    # 90' draw → extra time with halved draw mass, then penalties (50/50)
    ph, pd_, pa = p[0], p[1] * 0.5, p[2]
    tot = ph + pd_ + pa
    v = rng.random() * tot
    if v < ph:
        return home
    if v >= ph + pd_:
        return away
    return home if rng.random() < 0.5 else away


def draw_bracket(qualifiers: list[str], team_group: dict[str, str],
                 rng: np.random.Generator) -> list[str]:
    """
    Random Round-of-32 draw over the 32 qualifiers, greedily avoiding a pairing
    of two teams from the same group. Returns a flat list of 32 teams in bracket
    order (adjacent pairs meet in R32, their winners then feed the tree).
    """
    pool = qualifiers[:]
    rng.shuffle(pool)
    used = [False] * len(pool)
    order = []
    for i in range(len(pool)):
        if used[i]:
            continue
        used[i] = True
        a = pool[i]
        b = None
        for j in range(i + 1, len(pool)):           # prefer a different-group opponent
            if not used[j] and team_group[pool[j]] != team_group[a]:
                b, used[j] = pool[j], True
                break
        if b is None:                               # fallback: any remaining team
            for j in range(i + 1, len(pool)):
                if not used[j]:
                    b, used[j] = pool[j], True
                    break
        order.extend((a, b))
    return order


# ── Simulation driver ─────────────────────────────────────────────────────────

def simulate(groups, gm, n, rng):
    letters, P, GF, GA = group_aggregates(gm, n)
    GD = GF - GA
    teams_by_group = {g: groups[g] for g in letters}
    team_group = {t: g for g, ts in groups.items() for t in ts}

    # Accumulators
    reached = {rr: Counter() for rr in ROUNDS}
    win_group  = Counter()
    finish_sum = defaultdict(int)            # sum of finishing positions (group)
    third_qualify = Counter()                # times each group's 3rd place advanced

    for s in range(n):
        winners, runners = [], []
        thirds = []   # (pts, gd, gf, team, group)
        for gi, g in enumerate(letters):
            p, gd, gf = P[gi, :, s], GD[gi, :, s], GF[gi, :, s]
            order = rank_group(p, gd, gf, gm[g], s, rng)
            teams = teams_by_group[g]
            winners.append(teams[order[0]])
            runners.append(teams[order[1]])
            win_group[teams[order[0]]] += 1
            for pos, t_idx in enumerate(order):
                finish_sum[teams[t_idx]] += pos + 1
            ti = order[2]
            thirds.append((int(p[ti]), int(gd[ti]), int(gf[ti]), teams[ti], g))

        # Exactly 2 auto-qualifiers per group + the best 8 of the 12 third-placed
        thirds.sort(key=lambda x: (x[0], x[1], x[2], rng.random()), reverse=True)
        best_thirds = [t[3] for t in thirds[:8]]
        for t in thirds[:8]:
            third_qualify[t[4]] += 1

        qualifiers = winners + runners + best_thirds      # 32 distinct teams
        # Invariant: exactly 32 distinct qualifiers (2 per group + 8 thirds),
        # so no team can occupy more than one bracket slot in a simulation.
        assert len(qualifiers) == 32 and len(set(qualifiers)) == 32
        for t in qualifiers:
            reached["R32"][t] += 1

        # Fresh random bracket each sim, then single-elimination.
        cur = draw_bracket(qualifiers, team_group, rng)
        for rnd, nmatches in KO_ROUNDS:
            reach_key = "Winner" if rnd == "Champion" else rnd
            nxt = []
            for k in range(nmatches):
                w = ko_winner(cur[2 * k], cur[2 * k + 1], rng)
                nxt.append(w)
                reached[reach_key][w] += 1
            cur = nxt

    return letters, P, GF, GA, GD, dict(reached=reached, win_group=win_group,
                                        finish_sum=finish_sum, third_qualify=third_qualify)


# ── Output assembly ───────────────────────────────────────────────────────────

def r(x, d=4):
    return round(float(x), d)


def write_outputs(groups, gm, letters, P, GF, GA, GD, acc, n):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    reached, win_group = acc["reached"], acc["win_group"]
    finish_sum = acc["finish_sum"]
    all_teams = [t for ts in groups.values() for t in ts]

    # ── Champion blend: 0.75 · simulation + 0.25 · market prior (computed once,
    #    then used as the single champion figure across every output file) ──────
    market, market_meta = load_market_prior()
    sim_win = {team: reached["Winner"][team] / n for team in all_teams}
    if market:
        raw = {t: MARKET_BLEND * market.get(t, 0.0) + (1 - MARKET_BLEND) * sim_win[t]
               for t in all_teams}
    else:
        raw = dict(sim_win)
    tot = sum(raw.values()) or 1.0
    blended = {t: raw[t] / tot for t in all_teams}          # renormalise to sum 1.0

    # A champion probability can never exceed the probability of reaching the
    # final, so cap the blended figure at each team's simulated final-reach.
    # (Only binds for minnows whose market title odds outrun their sim run.)
    final_reach = {t: reached["Final"][t] / n for t in all_teams}
    blended = {t: min(blended[t], final_reach[t]) for t in all_teams}

    def champion(team):
        """Single source of truth for a team's title probability (all files)."""
        return blended[team]

    # 1. group_standings.json
    standings = {}
    for gi, g in enumerate(letters):
        rows = []
        for li, team in enumerate(groups[g]):
            rows.append({
                "team": team,
                "expected_points": r(P[gi, li].mean(), 3),
                "expected_gd":     r(GD[gi, li].mean(), 3),
                "expected_gf":     r(GF[gi, li].mean(), 3),
                "win_group_prob":  r(win_group[team] / n),
                "advance_prob":    r(reached["R32"][team] / n),
                "avg_finish":      r(finish_sum[team] / n, 3),
            })
        rows.sort(key=lambda x: (-x["advance_prob"], -x["expected_points"]))
        standings[g] = rows
    (PROCESSED / "group_standings.json").write_text(
        json.dumps(standings, indent=2, ensure_ascii=False))

    # 2. match_predictions.json (the 72 group matches, model predictions)
    preds = []
    for g in letters:
        for m in gm[g]:
            ph, pd_, pa = m.probs
            home, away = groups[g][m.a], groups[g][m.b]
            res = cm.predict_match(home, away, neutral=True)
            preds.append({
                "group": g, "home": home, "away": away,
                "p_home_win": r(ph), "p_draw": r(pd_), "p_away_win": r(pa),
                "most_likely_result": LABELS[int(np.argmax(m.probs))],
                "expected_scoreline": f"{res['most_likely_score'][0]}-{res['most_likely_score'][1]}",
                "xg_home": res["xg_home"], "xg_away": res["xg_away"],
                "confidence": res["confidence"],
            })
    (PROCESSED / "match_predictions.json").write_text(
        json.dumps(preds, indent=2, ensure_ascii=False))

    # 3. knockout_probabilities.json (per-team round-reached distribution)
    #    "Winner" uses the blended champion figure (consistent with the other
    #    files); R32..Final remain pure simulation (the market prior is a
    #    champion-level signal only).
    def round_prob(team, rnd):
        return champion(team) if rnd == "Winner" else reached[rnd][team] / n
    ko = {team: {rnd: r(round_prob(team, rnd)) for rnd in ROUNDS} for team in all_teams}
    ko = dict(sorted(ko.items(), key=lambda kv: -kv[1]["Winner"]))
    (PROCESSED / "knockout_probabilities.json").write_text(
        json.dumps(ko, indent=2, ensure_ascii=False))

    # 4. tournament_winner_odds.json (the blended champion figure, with components)
    ranked = sorted(all_teams, key=lambda t: -blended[t])
    odds = []
    for team in ranked[:10]:
        p = blended[team]
        odds.append({
            "team": team, "win_prob": r(p),
            "decimal_odds": r(1 / p, 2) if p > 0 else None,
            "data_model_prob": r(sim_win[team]),
            "market_prob": r(market.get(team, 0.0)),
        })
    (PROCESSED / "tournament_winner_odds.json").write_text(json.dumps({
        "n_simulations": n,
        "blend_weight": MARKET_BLEND,
        "blend_description": (f"{int(100*(1-MARKET_BLEND))}% data-driven simulation + "
                              f"{int(100*MARKET_BLEND)}% bookmaker market prior, renormalised, "
                              f"then capped at each team's simulated P(reach final)"),
        "market_source": market_meta,
        "top_10": odds,
    }, indent=2, ensure_ascii=False))

    blended_top = [{"team": t, "win_prob": r(blended[t])} for t in ranked[:10]]

    # 5. bracket.json (per-round projection: teams most likely to reach each round)
    #    Bracket slots are drawn at random per simulation, so the meaningful
    #    quantity is each team's probability of reaching a given round. The
    #    "Winner" round uses the blended champion figure (consistent with the
    #    other files); earlier rounds are pure simulation.
    rounds_out = {}
    for rnd in ROUNDS:
        if rnd == "Winner":
            ordered = sorted(((champion(t), t) for t in all_teams), reverse=True)
            rounds_out[rnd] = [{"team": t, "prob": r(p)} for p, t in ordered if p > 0]
        else:
            rounds_out[rnd] = [{"team": t, "prob": r(c / n)}
                               for t, c in reached[rnd].most_common()]
    bracket_json = {
        "method": ("Random Round-of-32 draw per simulation (no same-group R32 "
                   "meeting); odds are marginalised over all possible brackets. "
                   f"'Winner' = {int(100*(1-MARKET_BLEND))}% simulation + "
                   f"{int(100*MARKET_BLEND)}% market prior (see tournament_winner_odds.json)."),
        "rounds": rounds_out,
    }
    (PROCESSED / "bracket.json").write_text(
        json.dumps(bracket_json, indent=2, ensure_ascii=False))

    return standings, odds, blended


def print_standings(standings, groups_to_show):
    for g in groups_to_show:
        rows = standings[g]
        s = sum(x["advance_prob"] for x in rows)
        log.info("\n  Group %s   (Σ advance = %.1f%%)", g, 100 * s)
        log.info("    %-18s %6s %6s %7s %8s %9s", "team", "adv%", "win%", "pts", "exp_gd", "avg_fin")
        for x in rows:
            log.info("    %-18s %6.1f %6.1f %7.2f %+8.2f %9.2f",
                     x["team"], 100 * x["advance_prob"], 100 * x["win_group_prob"],
                     x["expected_points"], x["expected_gd"], x["avg_finish"])


def print_summary(standings, odds, blended, acc, letters, n):
    log.info("\n%s  FINAL TOURNAMENT WINNER ODDS — top 10  %s", "=" * 12, "=" * 12)
    log.info("  (75%% data-driven simulation + 25%% bookmaker market prior)")
    log.info("    %-18s %8s %12s %10s", "team", "win%", "data-model%", "market%")
    for o in odds:
        log.info("    %-18s %7.1f%% %11.1f%% %9.1f%%",
                 o["team"], 100 * o["win_prob"], 100 * o["data_model_prob"], 100 * o["market_prob"])

    log.info("\n%s  GROUPS C, H, J STANDINGS  %s", "=" * 16, "=" * 16)
    print_standings(standings, ["C", "H", "J"])

    # ── Verification of the post-fix invariants/targets ───────────────────────
    reached = acc["reached"]
    log.info("\n%s  VERIFICATION  %s", "=" * 20, "=" * 20)
    group_sums = {g: 100 * sum(x["advance_prob"] for x in standings[g]) for g in letters}
    total_adv = sum(reached["R32"].values()) / n
    log.info("  Global advancers per sim = %.2f  (must be 32)", total_adv)
    log.info("  Per-group advance sums: %s",
             "  ".join(f"{g}:{group_sums[g]:.0f}%" for g in letters))
    log.info("  (each = 200%% + that group's 3rd-place qualification rate; "
             "average ~267%%, global Σ = 3200%%)")

    targets = {"Spain": (18, 25), "Brazil": (8, 12), "England": (7, 10), "Argentina": (8, 12)}
    log.info("  Final (blended) win-probability targets:")
    all_ok = True
    for team, (lo, hi) in targets.items():
        p = 100 * blended[team]
        ok = lo <= p <= hi
        all_ok &= ok
        log.info("    %-10s %5.1f%%   target %2d-%2d%%   %s", team, p, lo, hi, "✓" if ok else "✗")
    log.info("  No team in >1 R32 slot per sim: ✓ (asserted each simulation)")
    log.info("  ALL TARGETS MET: %s", "✓ YES" if all_ok else "✗ NO")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)
    log.info("Loading combined model + fixtures (n_sims=%d) ...", N_SIMS)
    cm._ensure_loaded()
    dc = cm.load_dixon_coles()

    groups = load_groups()
    validate_team_names(groups, dc)

    log.info("Precomputing 72 group-match distributions ...")
    gm = build_group_matches(groups, dc)
    for g in gm:
        for m in gm[g]:
            m.sample(rng, N_SIMS)

    log.info("Pre-warming knockout probability cache (all team pairs) ...")
    prewarm_ko_cache([t for ts in groups.values() for t in ts])

    log.info("Running %d simulations ...", N_SIMS)
    letters, P, GF, GA, GD, acc = simulate(groups, gm, N_SIMS, rng)

    log.info("Writing outputs to %s ...", PROCESSED)
    standings, odds, blended = write_outputs(groups, gm, letters, P, GF, GA, GD, acc, N_SIMS)
    print_summary(standings, odds, blended, acc, letters, N_SIMS)
    log.info("\nDone. 5 JSON artifacts written to %s", PROCESSED)


if __name__ == "__main__":
    main()
