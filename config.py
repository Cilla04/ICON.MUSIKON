"""
config.py  –  Costanti e configurazione globale di MUSIKON
"""
import os

BASE_DIR      = os.path.dirname(__file__)
DATA_PATH     = os.path.join(BASE_DIR, "data", "playlist_tracks.csv")
ONTO_OWL      = os.path.join(BASE_DIR, "ontology", "music.owl")
KB_PATH       = os.path.join(BASE_DIR, "kb", "music_kb.pl")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_FEATURES = [
    "danceability", "energy", "loudness_norm", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo_norm", "key_norm",
]

OWL_ATOMIC = [
    "HighEnergy", "LowEnergy",
    "HighValence", "LowValence",
    "HighDanceability", "LowDanceability",
    "Acoustic", "Electronic",
    "Instrumental", "Vocal",
    "Live", "Studio",
    "Loud", "Quiet",
    "FastTempo", "SlowTempo",
    "Speechy",
]
OWL_COMPLEX = [
    "HappyTrack", "SadTrack",
    "EnergeticDanceable", "MellowAcoustic",
    "TenseTrack", "ChillTrack",
    "IntenseVocal", "AcousticSlow",
]
OWL_FEATURES = OWL_ATOMIC + OWL_COMPLEX  # 25 feature booleane

LABEL_MAP = {
    "Mhe":                    0,
    "GoodVibes":              1,
    "OldVibes":               2,
    "Anime & Japanese":       3,
    "Old Italian good vibes": 4,
}
CLASS_NAMES = ["Malinconico", "GoodVibes", "OldVibes", "Intenso", "Rilassato"]

N_SPLITS        = 10
TEST_SIZE       = 0.30
RANDOM_STATE    = 42

ALPHA           = 1.0
UNLABELED_RATIO = 0.30
MAX_ITER_EM     = 50
TOL_EM          = 1e-4

BN_MAX_ITER   = 500
BN_PRIOR_TYPE = "K2"
