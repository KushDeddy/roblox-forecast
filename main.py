#!/usr/bin/env python3
"""
Roblox Up-and-Coming Forecast Engine
Analyzes hourly chart snapshots to predict top 10 position probabilities.

Scoring: log score (ln(prob_gave_to_actual)). Must beat house AI (-37.2).
Strategy: Ensemble of velocity extrapolation, momentum, player-count growth,
          and mean-reversion signals. Softmax calibration for well-calibrated probs.
"""

import os
import sys
import json
import math
import glob
from collections import defaultdict


def load_input_data():
    """Load all hourly chart snapshots from /input directory."""
    input_dir = "/input"
    snapshots = []

    patterns = [
        os.path.join(input_dir, "*.json"),
        os.path.join(input_dir, "*.jsonl"),
        os.path.join(input_dir, "snapshots", "*.json"),
        os.path.join(input_dir, "data", "*.json"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    if os.path.isdir(input_dir):
        for f in sorted(os.listdir(input_dir)):
            fp = os.path.join(input_dir, f)
            if os.path.isfile(fp) and (f.endswith(".json") or f.endswith(".jsonl")):
                if fp not in files:
                    files.append(fp)

    if not files:
        print(f"WARNING: No input files found in {input_dir}", file=sys.stderr)
        return []

    for filepath in sorted(files):
        try:
            with open(filepath, "r") as fh:
                content = fh.read().strip()
                if not content:
                    continue
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            snapshots.append(item)
                elif isinstance(data, str):
                    for line in data.split("\n"):
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            snapshots.append(obj)
                elif isinstance(data, dict):
                    snapshots.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"ERROR reading {filepath}: {e}", file=sys.stderr)
            continue

    return snapshots


def parse_snapshot(snapshot):
    """Parse a snapshot into structured game data."""
    games = {}

    if "games" in snapshot:
        game_list = snapshot["games"]
    elif "chart" in snapshot:
        game_list = snapshot["chart"]
    elif "top_games" in snapshot:
        game_list = snapshot["top_games"]
    elif "rankings" in snapshot:
        game_list = snapshot["rankings"]
    elif "data" in snapshot:
        game_list = snapshot["data"]
    else:
        game_list = [snapshot]

    time_str = snapshot.get("time", snapshot.get("timestamp", snapshot.get("date", "")))

    for entry in game_list:
        if not isinstance(entry, dict):
            continue

        name = entry.get("name", entry.get("game_name", entry.get("title", "")))
        if not name:
            continue

        pos = entry.get("position", entry.get("place", entry.get("rank", None)))
        if pos is not None:
            try:
                pos = int(pos)
            except (ValueError, TypeError):
                pos = None

        players = entry.get("players", entry.get("player_count", entry.get("active_players",
                   entry.get("visits", None))))
        if players is not None:
            try:
                players = float(players)
            except (ValueError, TypeError):
                players = None

        velocity = entry.get("velocity", entry.get("movement", entry.get("change", None)))
        if velocity is not None:
            try:
                velocity = float(velocity)
            except (ValueError, TypeError):
                velocity = None

        games[name] = {
            "position": pos,
            "players": players,
            "velocity": velocity,
            "time": time_str,
            "raw": entry
        }

    return games, time_str


def compute_features(all_snapshots):
    """Compute forecasting features from all snapshots."""
    game_history = defaultdict(list)

    for snap_idx, snapshot in enumerate(all_snapshots):
        games, time_str = parse_snapshot(snapshot)
        for name, data in games.items():
            game_history[name].append({
                "snap_idx": snap_idx,
                "position": data["position"],
                "players": data["players"],
                "velocity": data["velocity"],
                "time": data["time"]
            })

    features = {}
    for name, history in game_history.items():
        positions = [h["position"] for h in history if h["position"] is not None]
        players_list = [h["players"] for h in history if h["players"] is not None]
        velocities = [h["velocity"] for h in history if h["velocity"] is not None]

        n = len(positions)
        if n == 0:
            continue

        current_pos = positions[-1]

        # Position trend
        if n >= 2:
            recent_avg = sum(positions[-min(3, n):]) / min(3, n)
            older_avg = sum(positions[:max(1, n//2)]) / max(1, n//2)
            trend = recent_avg - older_avg
        else:
            trend = 0.0

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        recent_velocity = velocities[-1] if velocities else 0.0

        player_growth = 0.0
        if len(players_list) >= 2:
            old_players = players_list[0]
            new_players = players_list[-1]
            if old_players > 0:
                player_growth = (new_players - old_players) / old_players

        recent_player_growth = 0.0
        if len(players_list) >= 3:
            mid_players = players_list[len(players_list)//2]
            if mid_players > 0:
                recent_player_growth = (players_list[-1] - mid_players) / mid_players

        if n >= 2:
            mean_pos = sum(positions) / n
            variance = sum((p - mean_pos) ** 2 for p in positions) / n
            std_pos = math.sqrt(variance)
        else:
            std_pos = 0.0

        weights = [2 ** i for i in range(n)]
        recency_weighted_pos = sum(positions[i] * weights[i] for i in range(n)) / sum(weights)

        mean_reversion_target = sum(positions) / n
        dist_to_top10 = max(0, current_pos - 10)

        features[name] = {
            "current_pos": current_pos,
            "trend": trend,
            "avg_velocity": avg_velocity,
            "recent_velocity": recent_velocity,
            "player_growth": player_growth,
            "recent_player_growth": recent_player_growth,
            "std_pos": std_pos,
            "recency_weighted_pos": recency_weighted_pos,
            "mean_reversion_target": mean_reversion_target,
            "dist_to_top10": dist_to_top10,
            "n_snapshots": n,
            "mean_position": sum(positions) / n,
            "min_position": min(positions),
            "max_position": max(positions),
        }

    return features


def predict_position(name, feat):
    """Predict future position for a game using ensemble signals."""
    scores = []
    weights = []

    # Signal 1: Velocity extrapolation
    if feat["avg_velocity"] is not None and abs(feat["avg_velocity"]) > 0.01:
        projected_pos = feat["current_pos"] + feat["avg_velocity"] * 3
        projected_pos = max(1, min(50, projected_pos))
        scores.append(projected_pos)
        weights.append(0.25)

    # Signal 2: Recency-weighted position
    scores.append(feat["recency_weighted_pos"])
    weights.append(0.30)

    # Signal 3: Mean reversion
    scores.append(feat["mean_reversion_target"])
    weights.append(0.15)

    # Signal 4: Player count growth
    if feat["player_growth"] != 0:
        player_signal = feat["current_pos"] - feat["player_growth"] * 25
        player_signal = max(1, min(50, player_signal))
        scores.append(player_signal)
        weights.append(0.20)

    # Signal 5: Recent trend
    trend_signal = feat["current_pos"] + feat["trend"] * 0.5
    trend_signal = max(1, min(50, trend_signal))
    scores.append(trend_signal)
    weights.append(0.10)

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_w = sum(weights)
    return weighted_sum / total_w


def generate_forecast(features):
    """Generate probability distribution over top 10 positions."""
    predictions = []
    for name, feat in features.items():
        pred_pos = predict_position(name, feat)
        predictions.append((name, pred_pos, feat))

    predictions.sort(key=lambda x: x[1])
    top_candidates = predictions[:min(30, len(predictions))]

    game_distributions = {}
    for name, pred_pos, feat in top_candidates:
        width = max(1.5, feat["std_pos"] * 1.2 + 1.0)

        dist = {}
        total_prob = 0
        for pos in range(1, 21):
            z = (pos - pred_pos) / width
            prob = math.exp(-0.5 * z * z)
            dist[pos] = prob
            total_prob += prob

        for pos in dist:
            dist[pos] /= total_prob

        game_distributions[name] = dist

    result = {}
    for name, pred_pos, feat in top_candidates:
        pos_probs = {}
        for pos in range(1, 11):
            pos_probs[str(pos)] = round(game_distributions[name].get(pos, 0.0), 6)
        result[name] = pos_probs

    top10_order = [p[0] for p in top_candidates[:10]]
    return result, top10_order


def calibrate_probabilities(forecast, top10_order):
    """Calibrate probabilities to be well-calibrated (not too extreme)."""
    calibrated = {}
    for name, pos_probs in forecast.items():
        raw_probs = [pos_probs[str(p)] for p in range(1, 11)]
        alpha = 0.15
        uniform = [1.0 / 10] * 10
        smoothed = [(1 - alpha) * r + alpha * u for r, u in zip(raw_probs, uniform)]
        total = sum(smoothed)
        normalized = [s / total for s in smoothed]
        calibrated[name] = {str(p + 1): round(n, 6) for p, n in enumerate(normalized)}
    return calibrated


def main():
    snapshots = load_input_data()
    if not snapshots:
        print("ERROR: No data loaded", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(snapshots)} snapshots", file=sys.stderr)

    features = compute_features(snapshots)
    print(f"Found {len(features)} games", file=sys.stderr)

    if len(features) < 2:
        print("ERROR: Not enough games found", file=sys.stderr)
        sys.exit(1)

    forecast, top10_order = generate_forecast(features)
    calibrated = calibrate_probabilities(forecast, top10_order)

    output = {
        "forecast": calibrated,
        "top_10_prediction": top10_order,
        "confidence": "medium",
        "method": "ensemble_velocity_momentum_player_growth_mean_reversion",
        "n_snapshots_used": len(snapshots),
        "n_games_analyzed": len(features)
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/forecast.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote forecast for {len(calibrated)} games", file=sys.stderr)
    print(f"Top 10: {top10_order}", file=sys.stderr)


if __name__ == "__main__":
    main()
