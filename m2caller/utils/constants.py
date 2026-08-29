# Token set: blank + A C G T
BLANK = 0
TOKENS = ["-", "A", "C", "G", "T"]
VOCAB_SIZE = len(TOKENS)

# Padding for signal (not used in data collate; kept for compatibility)
SIG_PAD = 0.0
