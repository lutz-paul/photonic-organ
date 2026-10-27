import wave
import numpy as np
from ml_dtypes import bfloat16
from qant_native_computing_toolkit.ai import calc_kan_layer_fprop
from qant_native_computing_toolkit.native import calc_scaled_periodic_nl_fprop


# SETTINGS
FS = 44100

# drawbars
KS = np.array([1, 3, 2, 4, 6, 8, 10, 12, 16], dtype=float)

METHOD = "single"      # "exact" / "kan" / "single" - main program loops over all three

# NOTES
NOTE_NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
              "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

def midi(note):
    #MIDI number, 69 = A4 = 440 Hz
    return 12 * (int(note[-1]) + 1) + NOTE_NAMES[note[:-1]]

def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def note_to_hz(note):
    return midi_to_hz(midi(note))


# ORGAN SOUND 
#
# Two ways to compute the same organ tone on the chip:
#
#   "kan":    organ_kan()    - ONE call of the KAN layer. The chip receives phase x
#                              rounded to bfloat16 and computes k*x itself.
#                              The rounding error of x gets multiplied by k
#                              -> high drawbars get noisy.
#                              Chip also does superposition of the harmonics:
#                              it adds up A * tcos(k*x) of all drawbars into one tone.
#
#   "single": organ_single() - One call of calc_scaled_periodic_nl_fprop per drawbar.
#                              The CPU computes k*x beforehand at full precision.
#                              Chip only computes tcos(u) * v.
#                              -> up to 9 calls per tone
#
#   "exact": organ_exact()   - the same tone without SDK, only used as reference.

def wrap_phase(phase):
    # phase to [-pi, pi)
    return (phase + np.pi) % (2 * np.pi) - np.pi


def organ_kan(x, levels):
    # phase already wrapped
    y = calc_kan_layer_fprop(
        x.reshape(-1, 1).astype(bfloat16),
        np.zeros((1, 1, 9)).astype(bfloat16),
        levels.reshape(1, 1, 9).astype(bfloat16),
        KS.astype(bfloat16),
    )
    return y.astype(np.float64).ravel()


def organ_single(phase, levels):
    # phase not wrapped yet
    y = np.zeros(len(phase))
    for a, k in zip(levels, KS):
        if a > 0: 
            u = wrap_phase(k * phase).astype(bfloat16)
            v = np.full(len(phase), a).astype(bfloat16)
            y += calc_scaled_periodic_nl_fprop(u, v).astype(np.float64)
    return y


def organ_exact(phase, levels):
    #Reference without SDK: the same tone computed exactly (float64, np.cos)
    return sum(a * np.cos(k * phase) for a, k in zip(levels, KS) if a > 0)


def organ_tone(phase, levels):
    if METHOD == "kan":
        return organ_kan(wrap_phase(phase), levels)
    if METHOD == "exact":
        return organ_exact(phase, levels)
    return organ_single(phase, levels)


def registration_to_levels(r):
    #Drawbar digits -> amplitudes, e.g. "008866432" -> [0, 0, 1, 1, 0.75, ...] (digit / 8)
    return np.array([int(c) for c in r], dtype=float) / 8.0


def envelope(n, fs, attack_ms, release_ms):
    # simulate attack and release, so notes don't click
    # Returns one volume factor (0 to 1) per sample.
    # Goes up at the start, stays at 1, goes down at the end.

    h = np.ones(n)

    n_attack = min(int(attack_ms / 1000 * fs), n)
    n_release = min(int(release_ms / 1000 * fs), n)

    if n_attack > 0:
        h[:n_attack] = np.linspace(0, 1, n_attack)

    if n_release > 0:
        h[-n_release:] = np.minimum(
            h[-n_release:],
            np.linspace(1, 0, n_release),
        )

    return h


# SHARED MELODY
# Both styles play exactly these notes: (note, duration in beats).
# An event is: (start beat, note, duration in beats, level)

# Ode an die Freude
MELODY = [
    ("E4", 1), ("E4", 1), ("F4", 1), ("G4", 1),
    ("G4", 1), ("F4", 1), ("E4", 1), ("D4", 1),
    ("C4", 1), ("C4", 1), ("D4", 1), ("E4", 1),
    ("E4", 1.5), ("D4", 0.5), ("D4", 2),
    ("E4", 1), ("E4", 1), ("F4", 1), ("G4", 1),
    ("G4", 1), ("F4", 1), ("E4", 1), ("D4", 1),
    ("C4", 1), ("C4", 1), ("D4", 1), ("E4", 1),
    ("D4", 1.5), ("C4", 0.5), ("C4", 2),
]

def sequence_events(sequence, level=1.0):
    events = []
    beat = 0.0

    for note, duration in sequence:
        events.append((beat, note, duration, level))
        beat += duration

    return events, beat


# STYLE: CLASSIC

CLASSIC = {
    "description": "Classic Organ",
    "tempo_bpm": 100,
    "registration": "808866432",
    "attack_ms": 12,
    "release_ms": 25,
}

# One chord per bar (4 beats)
CLASSIC_CHORDS = [
    ["C3", "G3", "E4", "G4"],
    ["F2", "C3", "A3", "F4"],
    ["C3", "G3", "E4", "G4"],
    ["G2", "D3", "B3", "G4"],
    ["C3", "G3", "E4", "G4"],
    ["F2", "C3", "A3", "F4"],
    ["G2", "D3", "B3", "F4"],
    ["C3", "G3", "E4", "C4"],
]


def classic_arrangement():
    melody, total_beats = sequence_events(MELODY, level=0.85)
    chords = []

    for bar, chord in enumerate(CLASSIC_CHORDS):
        start = bar * 4.0
        for i, note in enumerate(chord):
            level = 0.45 if i == 0 else 0.22       # lowest chord note a bit louder
            chords.append((start, note, 4.0, level))

    return melody + chords, total_beats


def render_classic(style):
    events, total_beats = classic_arrangement()
    levels = registration_to_levels(style["registration"])
    tempo = style["tempo_bpm"]

    total_samples = int(total_beats * 60.0 / tempo * FS)
    mix = np.zeros(total_samples, dtype=np.float64)

    for start_beat, note, duration, level in events:
        start = int(start_beat * 60.0 / tempo * FS)
        n = int(duration * 60.0 / tempo * FS)

        if n <= 0 or start >= total_samples:
            continue

        n = min(n, total_samples - start)

        t = np.arange(n) / FS
        f16 = note_to_hz(note) / 2                 # 16' drawbar sounds one octave lower

        phase = 2 * np.pi * f16 * t
        signal = organ_tone(phase, levels)
        env = envelope(n, FS, style["attack_ms"], style["release_ms"])

        mix[start:start + n] += level * signal * env

    return mix


# STYLE: HAMMOND 

HAMMOND = {
    "description": "Hammond Organ",
    "tempo_bpm": 80,
    "registration": {"melody": "888800008", "chords": "806800000", "bass": "008800000"},
    "attack_ms": 4,
    "release_ms": 25,
}

# Harmony: major triads with plagal turns (C - F), a few seventh chords,
# and a short "Amen" (F - C) at the end, only in the accompaniment.
# (start beat, duration, root, chord type)
HAMMOND_CHORDS = [
    (0, 2, "C", ""), (2, 2, "F", ""),           # E E | F G
    (4, 2, "F", ""), (6, 2, "G", "7"),          # G F | E D
    (8, 2, "C", ""), (10, 2, "A", "m7"),        # C C | D E
    (12, 2, "D", "m7"), (14, 2, "G", "7"),      # E. D | D
    (16, 2, "C", ""), (18, 2, "F", ""),
    (20, 2, "F", ""), (22, 2, "G", "7"),
    (24, 2, "C", ""), (26, 2, "A", "m7"),
    (28, 1, "D", "m7"), (29, 1, "G", "7"), (30, 2, "C", ""),
    (32, 2, "F", ""), (34, 2, "C", ""),   # Amen
]

# Intervals of the chord types in semitones above the root
CHORD_TYPES = {
    "":     [0, 4, 7],          # major
    "m":    [0, 3, 7],          # minor
    "7":    [0, 4, 7, 10],      # dominant seventh
    "m7":   [0, 3, 7, 10],      # minor seventh
}


def voicing(root, chord_type, low=48, high=60):
    # Chord tones as MIDI numbers, all between 'low' and 'high' (below the melody)
    tones = []
    for iv in CHORD_TYPES[chord_type]:
        pc = (NOTE_NAMES[root] + iv) % 12
        m = low + (pc - low) % 12                    # first position from 'low' upwards
        if m >= high:
            m -= 12
        tones.append(m)
    return sorted(set(tones))


def bass_note(pc, low=40):
    # Pitch class -> bass note between E2 (40) and D#3 (51).
    return low + (pc - low) % 12


def hammond_arrangement():
    # Events here have a 5th field, the voice: "melody", "chords" or "bass".
    melody, _ = sequence_events(MELODY, level=0.85)
    events = [(start, midi(note), duration, level, "melody")
              for start, note, duration, level in melody]

    for start, duration, root, chord_type in HAMMOND_CHORDS:
        tones = voicing(root, chord_type, low=48, high=60)
        # Sustained chord over the whole duration, plus a light accent on 2 and 4
        for m in tones:
            events.append((start, m, duration, 0.13, "chords"))
        for b in range(int(duration)):
            if int(start + b) % 2 == 1:
                for m in tones:
                    events.append((start + b, m, 0.5, 0.07, "chords"))
        # Bass: root over the whole duration
        events.append((start, bass_note(NOTE_NAMES[root]), duration, 0.5, "bass"))

    total_beats = max(s + d for s, _, d, _, _ in events)
    return events, total_beats


def render_hammond(style):
    # Same principle as render_classic(), but each voice has its own drawbars.
    events, total_beats = hammond_arrangement()
    sec_per_beat = 60.0 / style["tempo_bpm"]
    total_samples = int(total_beats * sec_per_beat * FS) + FS // 2     # + 0.5 s for the end
    mix = np.zeros(total_samples)
    levels = {voice: registration_to_levels(r) for voice, r in style["registration"].items()}

    for start_beat, m, duration, level, voice in events:
        start = int(start_beat * sec_per_beat * FS)
        n = min(int(duration * sec_per_beat * FS), total_samples - start)
        if n <= 0:
            continue

        t = np.arange(n) / FS
        f16 = midi_to_hz(m) / 2                    # 16' drawbar sounds one octave lower

        phase = 2 * np.pi * f16 * t
        signal = organ_tone(phase, levels[voice])
        env = envelope(n, FS, style["attack_ms"], style["release_ms"])

        mix[start:start + n] += level * signal * env

    return mix


# STYLE LIST
# Name -> (settings, render function). A new style only needs one line here.

STYLES = {
    "classic": (CLASSIC, render_classic),
    "hammond": (HAMMOND, render_hammond),
}


def render(name):
    # Renders a style
    settings, render_function = STYLES[name]
    return render_function(settings)


# WAV FILES + ANALYSIS
#
# Analysis: signal-to-error ratio (SINAD) measured from the WAV files.
# Reference: the exact computation in float64 
# The finished WAV files are measured - on real Q.ANT hardware their
# noise and inaccuracy would therefore be measured automatically.
#
#   error = WAV file - reference
#   SINAD = 10 * log10(signal power / error power)   in dB

def save_wav(music, filename, factor=None):
    """Saves as a 16-bit WAV. Without factor: loudest point at 0.8.
    With factor: all variants of one style are scaled with THE SAME factor,
    so the WAV files can be compared directly afterwards."""
    if factor is None:
        peak = np.max(np.abs(music))
        factor = 0.8 / peak if peak > 0 else 1.0
    music = np.clip(music * factor, -1.0, 1.0)

    pcm = (music * 32767).astype(np.int16)

    with wave.open(filename, "wb") as f:
        f.setnchannels(1)        # mono
        f.setsampwidth(2)        # 2 Byte = 16 Bit pro Sample
        f.setframerate(FS)       # 44100 Samples pro Sekunde
        f.writeframes(pcm.tobytes())

    print(f"Saved: {filename} ({len(music) / FS:.1f} seconds)")


def read_wav(filename):
    with wave.open(filename, "rb") as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return data.astype(np.float64) / 32767


def sinad_db(signal, reference):
    n = min(len(signal), len(reference))
    signal, reference = signal[:n], reference[:n]
    # Match the volume, so that a pure volume difference does not count as error
    g = np.dot(signal, reference) / np.dot(signal, signal)
    error = g * signal - reference
    return 10 * np.log10(np.mean(reference ** 2) / np.mean(error ** 2))



# MAIN PROGRAM: choose style -> 3 WAV files -> analysis

if __name__ == "__main__":
    names = list(STYLES)
    style = ""
    while style not in names:
        style = input(f"Which style? ({' / '.join(names)}): ").strip().lower()

    methods = ("exact", "kan", "single")
    factor = None
    reference = None
    for METHOD in methods:
        music = render(style)
        if factor is None:                     # scaling and reference come from the exact version
            factor = 0.8 / np.max(np.abs(music))
            reference = music * factor
        save_wav(music, f"organ_{style}_{METHOD}.wav", factor)

    print(f"\nSignal-to-error ratio (SINAD) compared to the exact computation:")
    explanation = {
        "exact": "measurement limit: only the 16-bit rounding of the WAV file",
        "kan": "KAN layer, one SDK call per note",
        "single": "calc_scaled_periodic_nl_fprop once per drawbar",
    }
    for method in methods:
        value = sinad_db(read_wav(f"organ_{style}_{method}.wav"), reference)
        print(f"  {method:7s} {value:6.1f} dB   ({explanation[method]})")