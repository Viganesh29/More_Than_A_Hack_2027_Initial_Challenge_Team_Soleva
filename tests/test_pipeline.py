"""Tests for the possession pipeline. No competition data is used: tracking is built in memory or
written as a fake file in the official ATD layout (tests/fixtures/fake_atd.py)."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from src.atd_loader import parse_epts_metadata, read_epts_raw  # noqa: E402
from src.event_labels import labels_from_events  # noqa: E402
from src.params_loader import ParameterValidationError, load_params_tuned  # noqa: E402
from src.possession_model import PossessionConfig, predict_possession  # noqa: E402
from src.tracking import UnifiedTrackingData  # noqa: E402


# --------------------------------------------------------------------------- games.py

def test_games_matches_official_constants():
    assert games.GAMES["BAR-ATH"]["n_video_frames"] == 199319
    assert games.GAMES["BAR-ATH"]["pitch"] == (105.3, 68.0)
    assert games.half_video_blocks("BAR-ATH") == [(1, 11185, 90651), (2, 110808, 190197)]
    paths = games.asset_paths()
    assert paths["atd_raw"].endswith(os.path.join("tactical", "BAR-ATH_tactical_atd.txt"))
    assert paths["atd_meta"].endswith(os.path.join("tactical", "BAR-ATH_tactical_atd.xml"))


# --------------------------------------------------------------------------- ATD loader

XML_NO_SPEC = """<?xml version="1.0"?><main><Metadata>
<Teams><Team id="H1"><Name>Home</Name></Team><Team id="A1"><Name>Away</Name></Team></Teams>
<Players><Player id="p1" teamId="H1"/><Player id="p2" teamId="A1"/><Player id="p3" teamId="H1"/></Players>
</Metadata></main>"""

XML_WITH_SPEC = """<?xml version="1.0"?><main><Metadata>
<Teams><Team id="H1"/><Team id="A1"/></Teams>
<Players><Player id="p1" teamId="H1"/><Player id="p2" teamId="A1"/><Player id="p3" teamId="H1"/></Players>
<PlayerChannels>
 <PlayerChannel id="c1x" channelId="x" playerId="p1"/><PlayerChannel id="c1y" channelId="y" playerId="p1"/>
 <PlayerChannel id="c2x" channelId="x" playerId="p2"/><PlayerChannel id="c2y" channelId="y" playerId="p2"/>
 <PlayerChannel id="c3x" channelId="x" playerId="p3"/><PlayerChannel id="c3y" channelId="y" playerId="p3"/>
</PlayerChannels>
</Metadata>
<DataFormatSpecifications>
 <DataFormatSpecification startFrame="1" endFrame="2" separator=":">
  <PlayerChannelRef playerChannelId="c2x"/><PlayerChannelRef playerChannelId="c2y"/>
  <PlayerChannelRef playerChannelId="c1x"/><PlayerChannelRef playerChannelId="c1y"/>
 </DataFormatSpecification>
 <DataFormatSpecification startFrame="3" endFrame="3" separator=":">
  <PlayerChannelRef playerChannelId="c3x"/><PlayerChannelRef playerChannelId="c3y"/>
  <PlayerChannelRef playerChannelId="c2x"/><PlayerChannelRef playerChannelId="c2y"/>
 </DataFormatSpecification>
</DataFormatSpecifications></main>"""


def test_atd_without_spec_uses_player_order_and_frame_id_minus_one(tmp_path):
    meta = tmp_path / "m.xml"
    raw = tmp_path / "r.txt"
    meta.write_text(XML_NO_SPEC)
    raw.write_text(
        "1:0.1,0.2,1.0;NaN,NaN,NaN;0.3,0.4,2.0:0.5,0.5,0.0\n"
        "2:0.2,0.2,1.0;0.6,0.6,1.0;NaN:NaN,NaN,NaN\n"
    )
    layout = parse_epts_metadata(meta)
    assert layout.is_home.tolist() == [True, False, True]
    players, ball, present = read_epts_raw(raw, layout, n_frames=3)
    assert present.tolist() == [True, True, False]
    np.testing.assert_allclose(players[0, 0], [0.1, 0.2])  # frame_id 1 -> row 0
    assert np.isnan(players[0, 1]).all()
    np.testing.assert_allclose(players[1, 1], [0.6, 0.6])
    assert np.isnan(players[1, 2]).all()  # bare "NaN" group
    np.testing.assert_allclose(ball[0], [0.5, 0.5])
    assert np.isnan(ball[1]).all()


def test_atd_with_spec_maps_slots_through_channels(tmp_path):
    meta = tmp_path / "m.xml"
    raw = tmp_path / "r.txt"
    meta.write_text(XML_WITH_SPEC)
    raw.write_text("1:0.9,0.9;0.1,0.1:NaN,NaN\n3:0.3,0.3;0.8,0.8:0.5,0.5\n")
    layout = parse_epts_metadata(meta)
    players, _, _ = read_epts_raw(raw, layout, n_frames=3)
    np.testing.assert_allclose(players[0, 1], [0.9, 0.9])  # first slot of spec 1 is p2
    np.testing.assert_allclose(players[0, 0], [0.1, 0.1])  # second slot is p1
    np.testing.assert_allclose(players[2, 2], [0.3, 0.3])  # spec 2: p3 then p2
    np.testing.assert_allclose(players[2, 1], [0.8, 0.8])


def test_atd_slot_count_mismatch_raises(tmp_path):
    meta = tmp_path / "m.xml"
    raw = tmp_path / "r.txt"
    meta.write_text(XML_NO_SPEC)
    raw.write_text("1:0.1,0.2;0.3,0.4:0.5,0.5\n")
    with pytest.raises(ValueError):
        read_epts_raw(raw, parse_epts_metadata(meta), n_frames=1)


# --------------------------------------------------------------------------- engine rules

def make_tracking(n, home_xy, away_xy, ball_xy, periods=None):
    """Tracking with one home and one away player, coordinates in metres."""
    return UnifiedTrackingData(
        frame_indices=np.arange(n),
        periods=np.ones(n, dtype=np.int32) if periods is None else periods,
        home_positions=np.asarray(home_xy, dtype=np.float32).reshape(n, 1, 2),
        away_positions=np.asarray(away_xy, dtype=np.float32).reshape(n, 1, 2),
        ball_position=np.asarray(ball_xy, dtype=np.float32).reshape(n, 2),
    )


NO_STOPPAGE = {"enable_stoppage_speed": False, "min_spell_frames": 1, "enable_stoppage_bridging": False}


def scenario(n=300):
    home = np.tile([-10.0, 0.0], (n, 1))
    away = np.tile([10.0, 0.0], (n, 1))
    ball = home + [0.5, 0.0]
    return home, away, ball


def test_missing_ball_keeps_possession():
    home, away, ball = scenario()
    ball[100:180] = np.nan
    labels = predict_possession(make_tracking(300, home, away, ball), NO_STOPPAGE)
    assert set(labels[1:]) == {"H"}


def test_loose_ball_keeps_possession():
    home, away, ball = scenario()
    ball[100:150] = [0.0, 0.0]  # 10 m from both players, ball in flight
    labels = predict_possession(make_tracking(300, home, away, ball), NO_STOPPAGE)
    assert set(labels[1:]) == {"H"}


def test_change_needs_dwell():
    home, away, ball = scenario()
    ball[100:105] = away[100:105] - [0.5, 0.0]  # 5-frame touch by the opponent
    ball[200:] = away[200:] - [0.5, 0.0]  # sustained control by the opponent
    params = dict(NO_STOPPAGE, dwell_frames=10)
    labels = predict_possession(make_tracking(300, home, away, ball), params)
    assert set(labels[1:200]) == {"H"}
    assert labels[205] == "H" and labels[215] == "A" and labels[-1] == "A"


def test_ball_out_of_bounds_is_dead():
    home, away, ball = scenario()
    ball[100:150] = [0.0, 40.0]  # beyond the 34 m touchline
    labels = predict_possession(make_tracking(300, home, away, ball), NO_STOPPAGE)
    assert set(labels[100:150]) == {"D"}
    assert labels[99] == "H" and labels[160] == "H"


def test_long_missing_ball_becomes_dead():
    home, away, ball = scenario()
    ball[50:250] = np.nan
    params = dict(NO_STOPPAGE, max_missing_hold_frames=100)
    labels = predict_possession(make_tracking(300, home, away, ball), params)
    assert labels[100] == "H" and labels[200] == "D"


def test_static_players_are_a_stoppage():
    n = 400
    t = np.arange(n)
    home = np.stack([-10 + 3 * np.sin(t / 10.0), np.zeros(n)], axis=1)  # jogging
    away = np.stack([10 + 3 * np.sin(t / 10.0), np.zeros(n)], axis=1)
    home[150:300] = home[150]  # everyone stands still for 6 s
    away[150:300] = away[150]
    ball = home + [0.5, 0.0]
    params = {"stoppage_speed_mps": 1.0, "stoppage_min_frames": 50, "min_spell_frames": 1,
              "min_tracked_players": 1}
    labels = predict_possession(make_tracking(n, home, away, ball), params)
    assert labels[100] == "H"
    assert set(labels[170:280]) == {"D"}
    assert labels[350] == "H"


def test_period_zero_is_not_processed():
    home, away, ball = scenario()
    periods = np.ones(300, dtype=np.int32)
    periods[:50] = 0
    labels = predict_possession(make_tracking(300, home, away, ball, periods), NO_STOPPAGE)
    assert set(labels[:50]) == {"D"} and labels[100] == "H"


# --------------------------------------------------------------------------- params

def test_params_loader_needs_no_scores(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"parameters": {"dwell_frames": 12}}))
    assert load_params_tuned(path) == {"dwell_frames": 12}
    PossessionConfig.from_dict({"parameters": {"dwell_frames": 12}})


@pytest.mark.parametrize("bad", [{"nope": 1}, {"dwell_frames": 1.5}, {"distance_threshold": -1.0}])
def test_params_loader_rejects_bad_values(tmp_path, bad):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"parameters": bad}))
    with pytest.raises(ParameterValidationError):
        load_params_tuned(path)


def test_repository_params_file_loads():
    load_params_tuned(os.path.join(ROOT, "params_tuned.json"))


# --------------------------------------------------------------------------- event labels

def test_event_labels_rules():
    events = pd.DataFrame(
        [
            ("H", "SET PIECE", "KICK OFF", 1, 10, 10),
            ("H", "PASS", "", 1, 20, 30),
            ("H", "BALL LOST", "INTERCEPTION", 1, 40, 40),
            ("A", "RECOVERY", "INTERCEPTION", 1, 50, 50),
            ("A", "BALL OUT", "", 1, 80, 85),
            ("H", "SET PIECE", "THROW IN", 1, 120, 120),
            ("H", "PASS", "", 1, 125, 140),
        ],
        columns=["team", "type", "subtype", "period", "start_frame", "end_frame"],
    )
    labels = labels_from_events(events, 200)
    # row r = frame r + 1
    assert set(labels[:9]) == {"X"}
    assert set(labels[9:49]) == {"H"}  # frames 10-49: home, the loose ball after BALL LOST stays home
    assert set(labels[49:84]) == {"A"}  # frames 50-84
    assert set(labels[84:119]) == {"D"}  # frames 85-119: out until the throw-in
    assert set(labels[119:140]) == {"H"}
    assert set(labels[140:]) == {"X"}


# --------------------------------------------------------------------------- pipeline

def test_pipeline_fails_loudly_without_data(tmp_path):
    from run_pipeline import run_pipeline

    with pytest.raises(IOError):
        run_pipeline(tmp_path / "out.csv", data_dir=str(tmp_path / "missing"), quiet=True)


@pytest.mark.slow
def test_pipeline_end_to_end_on_fake_atd(tmp_path):
    from run_pipeline import run_pipeline
    from tests.fixtures.fake_atd import write_fake_atd

    write_fake_atd(str(tmp_path / "data"))
    out = run_pipeline(tmp_path / "fc.csv", data_dir=str(tmp_path / "data"), quiet=True)
    table = pm.read_label_csv(out)
    assert pm.validate_submission(table, 199319) is None
    labels = table["label"].to_numpy()
    assert set(labels[:11185]) == {"X"}
    assert set(labels[90652:110808]) == {"X"}
    assert set(labels[190198:]) == {"X"}
    graded = np.r_[labels[11185:90652], labels[110808:190198]]
    assert set(graded) <= {"H", "A", "D"}
    assert {"H", "A", "D"} <= set(graded)
