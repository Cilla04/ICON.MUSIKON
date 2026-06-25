"""
build_ontology.py
-----------------
Genera l'ontologia OWL per il dominio musicale usando owlready2.
Definisce classi per feature audio di Spotify e concetti derivati
utilizzati come Background Knowledge (BK) per il sistema ML.

Università degli Studi di Bari Aldo Moro - ICon 2023/2024
"""

from owlready2 import (
    get_ontology, Thing, DataProperty, ObjectProperty,
    ConstrainedDatatype, FunctionalProperty,
)
import types
import os

ONTO_IRI = "http://musikon.uniba.it/music.owl#"

def build_music_ontology(output_path: str) -> None:
    """
    Costruisce l'ontologia OWL per il dominio musicale.
    
    Classi primitive:
      - MusicTrack: entità principale (traccia musicale)
      - MoodCategory: categorie di umore (PlaylistMhe, PlaylistGoodVibes, ...)
    
    Data properties (feature audio Spotify):
      - hasDanceability, hasEnergy, hasValence, hasAcousticness,
        hasInstrumentalness, hasLiveness, hasLoudness (normalizzata),
        hasSpeechiness, hasTempo (normalizzato), hasKey
    
    Classi derivate (concetti atomici su soglie):
      - HighEnergy, LowEnergy, HighValence, LowValence,
        HighDanceability, LowDanceability, Acoustic, Electronic,
        Instrumental, Vocal, Live, Studio, Loud, Quiet,
        FastTempo, SlowTempo, Speechy
    
    Concetti complessi (intersezione di atomici):
      - HappyTrack, SadTrack, EnergeticDanceable, MellowAcoustic,
        TenseTrack, ChillTrack
    """
    onto = get_ontology(ONTO_IRI)

    with onto:
        # ─── Classi principali ───────────────────────────────────────
        class MusicTrack(Thing): pass
        class MoodCategory(Thing): pass

        class PlaylistMhe(MoodCategory): pass            # Malinconico
        class PlaylistGoodVibes(MoodCategory): pass      # Positivo
        class PlaylistOldVibes(MoodCategory): pass       # Nostalgico
        class PlaylistAnime(MoodCategory): pass          # Intenso/Energico
        class PlaylistItalian(MoodCategory): pass        # Rilassato/Acustico

        # ─── Data Properties (feature audio) ─────────────────────────
        class hasDanceability(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasEnergy(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasValence(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasAcousticness(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasInstrumentalness(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasLiveness(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasNormLoudness(DataProperty, FunctionalProperty):
            """Loudness normalizzata in [0,1]: (dB - min) / (max - min)"""
            domain = [MusicTrack]
            range  = [float]

        class hasSpeechiness(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [float]

        class hasNormTempo(DataProperty, FunctionalProperty):
            """Tempo normalizzato in [0,1]: (bpm - 54) / (207 - 54)"""
            domain = [MusicTrack]
            range  = [float]

        class hasKey(DataProperty, FunctionalProperty):
            domain = [MusicTrack]
            range  = [int]

        # ─── Classi derivate ATOMICHE (per soglie) ──────────────────
        # Queste classi rappresentano le condizioni logiche sulle feature.
        # Le soglie sono state scelte in base ai quartili del dataset.
        # Corrispondono ai "feature concepts" usati in MBM-EM.

        class HighEnergy(MusicTrack):
            """energy > 0.75 (75° percentile dataset)"""
            equivalent_to = [MusicTrack & hasEnergy.value(0.75)]

        class LowEnergy(MusicTrack):
            """energy < 0.45 (25° percentile circa)"""
            equivalent_to = [MusicTrack & hasEnergy.value(0.45)]

        class HighValence(MusicTrack):
            """valence > 0.65"""
            equivalent_to = [MusicTrack & hasValence.value(0.65)]

        class LowValence(MusicTrack):
            """valence < 0.32"""
            equivalent_to = [MusicTrack & hasValence.value(0.32)]

        class HighDanceability(MusicTrack):
            """danceability > 0.68"""
            equivalent_to = [MusicTrack & hasDanceability.value(0.68)]

        class LowDanceability(MusicTrack):
            """danceability < 0.47"""
            equivalent_to = [MusicTrack & hasDanceability.value(0.47)]

        class Acoustic(MusicTrack):
            """acousticness > 0.40"""
            equivalent_to = [MusicTrack & hasAcousticness.value(0.40)]

        class Electronic(MusicTrack):
            """acousticness < 0.05"""
            equivalent_to = [MusicTrack & hasAcousticness.value(0.05)]

        class Instrumental(MusicTrack):
            """instrumentalness > 0.05"""
            equivalent_to = [MusicTrack & hasInstrumentalness.value(0.05)]

        class Vocal(MusicTrack):
            """instrumentalness < 0.001"""
            equivalent_to = [MusicTrack & hasInstrumentalness.value(0.001)]

        class Live(MusicTrack):
            """liveness > 0.35"""
            equivalent_to = [MusicTrack & hasLiveness.value(0.35)]

        class Studio(MusicTrack):
            """liveness < 0.12"""
            equivalent_to = [MusicTrack & hasLiveness.value(0.12)]

        class Loud(MusicTrack):
            """norm_loudness > 0.72"""
            equivalent_to = [MusicTrack & hasNormLoudness.value(0.72)]

        class Quiet(MusicTrack):
            """norm_loudness < 0.40"""
            equivalent_to = [MusicTrack & hasNormLoudness.value(0.40)]

        class FastTempo(MusicTrack):
            """norm_tempo > 0.55 (~138 bpm)"""
            equivalent_to = [MusicTrack & hasNormTempo.value(0.55)]

        class SlowTempo(MusicTrack):
            """norm_tempo < 0.30 (~100 bpm)"""
            equivalent_to = [MusicTrack & hasNormTempo.value(0.30)]

        class Speechy(MusicTrack):
            """speechiness > 0.07"""
            equivalent_to = [MusicTrack & hasSpeechiness.value(0.07)]

        # ─── Classi derivate COMPLESSE (intersezioni — ragionamento OWL) ─
        class HappyTrack(MusicTrack):
            """Traccia Happy: HighValence ⊓ HighEnergy"""
            equivalent_to = [HighValence & HighEnergy]

        class SadTrack(MusicTrack):
            """Traccia Sad: LowValence ⊓ LowEnergy ⊓ Acoustic"""
            equivalent_to = [LowValence & LowEnergy & Acoustic]

        class EnergeticDanceable(MusicTrack):
            """Energica/Danzante: HighEnergy ⊓ HighDanceability"""
            equivalent_to = [HighEnergy & HighDanceability]

        class MellowAcoustic(MusicTrack):
            """Mellow/Introspettiva: LowEnergy ⊓ Acoustic"""
            equivalent_to = [LowEnergy & Acoustic]

        class TenseTrack(MusicTrack):
            """Tesa/Angosciata: LowValence ⊓ HighEnergy"""
            equivalent_to = [LowValence & HighEnergy]

        class ChillTrack(MusicTrack):
            """Rilassata: HighValence ⊓ LowEnergy"""
            equivalent_to = [HighValence & LowEnergy]

        class IntenseVocal(MusicTrack):
            """Intensa/Vocale: HighEnergy ⊓ Vocal ⊓ Electronic"""
            equivalent_to = [HighEnergy & Vocal & Electronic]

        class AcousticSlow(MusicTrack):
            """Ballata acustica: Acoustic ⊓ SlowTempo"""
            equivalent_to = [Acoustic & SlowTempo]

    # Salva ontologia
    onto.save(file=output_path, format="rdfxml")
    print(f"[OK] Ontologia salvata in: {output_path}")
    print(f"     Classi definite: {len(list(onto.classes()))}")
    print(f"     Data properties: {len(list(onto.data_properties()))}")


# ─── Mappa soglie (sincronizzata con l'ontologia) ──────────────────────────
# Usata da ontology_features.py per il ragionamento senza Pellet
FEATURE_THRESHOLDS = {
    # Atomici
    "HighEnergy":        lambda r: r["energy"]            > 0.75,
    "LowEnergy":         lambda r: r["energy"]            < 0.45,
    "HighValence":       lambda r: r["valence"]           > 0.65,
    "LowValence":        lambda r: r["valence"]           < 0.32,
    "HighDanceability":  lambda r: r["danceability"]      > 0.68,
    "LowDanceability":   lambda r: r["danceability"]      < 0.47,
    "Acoustic":          lambda r: r["acousticness"]      > 0.40,
    "Electronic":        lambda r: r["acousticness"]      < 0.05,
    "Instrumental":      lambda r: r["instrumentalness"]  > 0.05,
    "Vocal":             lambda r: r["instrumentalness"]  < 0.001,
    "Live":              lambda r: r["liveness"]          > 0.35,
    "Studio":            lambda r: r["liveness"]          < 0.12,
    "Loud":              lambda r: r["norm_loudness"]     > 0.72,
    "Quiet":             lambda r: r["norm_loudness"]     < 0.40,
    "FastTempo":         lambda r: r["norm_tempo"]        > 0.55,
    "SlowTempo":         lambda r: r["norm_tempo"]        < 0.30,
    "Speechy":           lambda r: r["speechiness"]       > 0.07,
    # Complessi (= AND di atomici — derivati dal ragionamento OWL)
    "HappyTrack":        lambda r: r["valence"]    > 0.65 and r["energy"]       > 0.75,
    "SadTrack":          lambda r: r["valence"]    < 0.32 and r["energy"]       < 0.45 and r["acousticness"] > 0.40,
    "EnergeticDanceable":lambda r: r["energy"]     > 0.75 and r["danceability"] > 0.68,
    "MellowAcoustic":    lambda r: r["energy"]     < 0.45 and r["acousticness"] > 0.40,
    "TenseTrack":        lambda r: r["valence"]    < 0.32 and r["energy"]       > 0.75,
    "ChillTrack":        lambda r: r["valence"]    > 0.65 and r["energy"]       < 0.45,
    "IntenseVocal":      lambda r: r["energy"]     > 0.75 and r["instrumentalness"] < 0.001 and r["acousticness"] < 0.05,
    "AcousticSlow":      lambda r: r["acousticness"] > 0.40 and r["norm_tempo"] < 0.30,
}

COMPLEX_CONCEPTS = ["HappyTrack", "SadTrack", "EnergeticDanceable",
                    "MellowAcoustic", "TenseTrack", "ChillTrack",
                    "IntenseVocal", "AcousticSlow"]

ATOMIC_CONCEPTS = [k for k in FEATURE_THRESHOLDS if k not in COMPLEX_CONCEPTS]


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "music.owl")
    build_music_ontology(out)
