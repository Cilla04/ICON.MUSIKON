%% music_kb.pl  –  Knowledge Base Prolog per MUSIKON
%%
%% Adattato da: Fonty02/ICON (kb.pl)
%%
%% La KB codifica regole logiche sul dominio musicale.
%% Integra conoscenza di background (BK) coerente con l'ontologia OWL.
%%
%% Utilizzo in Python tramite pyswip:
%%   prolog = Prolog()
%%   prolog.consult("kb/music_kb.pl")
%%   list(prolog.query("mood(Track, M), M = malinconico"))
%%
%% Struttura:
%%   1. Fatti base (asseriti dinamicamente dal modulo Python)
%%   2. Regole di classificazione mood
%%   3. Regole di validazione / rilevamento conflitti
%%   4. Regole di arricchimento feature

:- discontiguous feature/3.
:- discontiguous mood_predicted/2.
:- discontiguous conflict/2.

%% ─── 1. CLASSIFICAZIONE MOOD ────────────────────────────────────────────────
%% Le regole seguenti rispecchiano le definizioni OWL di concetti complessi.
%% Ordine: più specifico → meno specifico (Prolog usa first-match).

%% Malinconico: bassa energia, bassa valenza, spesso acustico
mood(Track, malinconico) :-
    feature(Track, energy,       E), E < 0.45,
    feature(Track, valence,      V), V < 0.32,
    feature(Track, acousticness, A), A > 0.40.

%% Malinconico alternativo: bassa energia e bassa valenza (senza acustico obbligatorio)
mood(Track, malinconico) :-
    feature(Track, energy,  E), E < 0.40,
    feature(Track, valence, V), V < 0.28,
    \+ mood(Track, intenso).

%% Intenso: alta energia, alta danzabilità, bassa valenza o non acustico
mood(Track, intenso) :-
    feature(Track, energy,       E), E > 0.75,
    feature(Track, danceability, D), D > 0.65,
    feature(Track, acousticness, A), A < 0.10.

%% Intenso alternativo: alta energia e veloce
mood(Track, intenso) :-
    feature(Track, energy,    E), E > 0.80,
    feature(Track, tempo_norm,T), T > 0.55.

%% GoodVibes: alta valenza, alta danzabilità, energia medio-alta
mood(Track, goodvibes) :-
    feature(Track, valence,      V), V > 0.65,
    feature(Track, energy,       E), E > 0.55,
    feature(Track, danceability, D), D > 0.60.

%% Rilassato: alta valenza, bassa energia (chill/calmo)
mood(Track, rilassato) :-
    feature(Track, valence, V), V > 0.55,
    feature(Track, energy,  E), E < 0.50,
    feature(Track, tempo_norm, T), T < 0.45.

%% Rilassato acustico (OldItalian): acustico + lento + non troppo triste
mood(Track, rilassato) :-
    feature(Track, acousticness, A), A > 0.50,
    feature(Track, energy,       E), E < 0.55,
    feature(Track, valence,      V), V > 0.30.

%% OldVibes: caratteristiche intermedie/miste → via esclusione
mood(Track, oldvibes) :-
    \+ mood(Track, malinconico),
    \+ mood(Track, intenso),
    \+ mood(Track, goodvibes),
    \+ mood(Track, rilassato),
    feature(Track, energy, E), E > 0.30, E < 0.80.

%% ─── 2. PROPRIETA' DERIVATE (coerenti con OWL BK) ──────────────────────────

is_acoustic(Track)     :- feature(Track, acousticness, A),     A > 0.40.
is_instrumental(Track) :- feature(Track, instrumentalness, I), I > 0.05.
is_live(Track)         :- feature(Track, liveness, L),         L > 0.35.
is_fast(Track)         :- feature(Track, tempo_norm, T),       T > 0.55.
is_slow(Track)         :- feature(Track, tempo_norm, T),       T < 0.30.
is_loud(Track)         :- feature(Track, loudness_norm, L),    L > 0.72.
is_speechy(Track)      :- feature(Track, speechiness, S),      S > 0.07.

high_energy(Track)     :- feature(Track, energy,       E), E > 0.75.
low_energy(Track)      :- feature(Track, energy,       E), E < 0.45.
high_valence(Track)    :- feature(Track, valence,      V), V > 0.65.
low_valence(Track)     :- feature(Track, valence,      V), V < 0.32.
high_dance(Track)      :- feature(Track, danceability, D), D > 0.68.

%% ─── 3. REGOLE DI VALIDAZIONE / CONFLITTO ───────────────────────────────────

%% Conflitto: una traccia non può essere simultaneamente malinconico e goodvibes
conflict(Track, mood_conflict) :-
    mood(Track, malinconico),
    mood(Track, goodvibes).

%% Conflitto: traccia acustica con alta energia elettronica
conflict(Track, acoustic_electric_conflict) :-
    feature(Track, acousticness,  A), A > 0.70,
    feature(Track, instrumentalness, I), I < 0.001,
    feature(Track, energy, E), E > 0.85.

%% ─── 4. QUERY DI UTILITA' ────────────────────────────────────────────────────

%% Elenca tutti i mood predetti per una traccia (senza duplicati)
all_moods(Track, Moods) :-
    findall(M, mood(Track, M), MoodList),
    sort(MoodList, Moods).

%% Mood prevalente (primo nella gerarchia di specificità)
primary_mood(Track, Mood) :-
    mood(Track, Mood), !.

%% Verifica se una traccia ha conflitti
has_conflict(Track) :-
    conflict(Track, _).
