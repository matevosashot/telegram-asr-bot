"""Fixed model constants (Voxtral Mini 4B Realtime)."""

SAMPLE_RATE = 16000
HOP_LENGTH = 160          # mel hop (10 ms)
WINDOW_SIZE = 400         # mel window (25 ms)
NUM_MEL_BINS = 128
GLOBAL_LOG_MEL_MAX = 1.5
SAMPLES_PER_FRAME = 1280  # one decoder frame = 80 ms
FRAME_MS = 80

TOKEN_BOS = 1
TOKEN_EOS = 2
TOKEN_PAD = 32            # "nothing to emit yet"
TOKEN_WORD = 33
NON_TEXT_TOKENS = frozenset((TOKEN_EOS, TOKEN_PAD, TOKEN_WORD))

LEFT_PAD_FRAMES = 32
RIGHT_PAD_BUFFER_FRAMES = 10   # extra silence after the delay so long words can finish
DEFAULT_DELAY_MS = 2400        # offline: longest delay (the model's own streaming default is 480 ms)
MIN_DELAY_FRAMES = 1
MAX_DELAY_FRAMES = 30          # 80 ms .. 2.4 s
