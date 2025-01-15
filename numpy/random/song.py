import sys
from time import sleep

def print_lyrics():
    lines = [
        ("So I'ma love you every night like it's the last night", 0.10),
        ("Like it's the last night", 0.10),
        ("If the world was ending", 0.10),
        ("I'd wanna be next to you", 0.10),
        ("If the party was over", 0.10),
        ("And our time on Earth was through", 0.10),
        ("I'd wanna hold you just for a while", 0.10),
        ("And die with a smile", 0.10),
        ("If the world was ending", 0.10),
        ("I'd wanna be next to you", 0.10),
        ("Right next to you", 0.10),
    ]

    delays = [0.6, 0.7, 1.0, 4.6, 1.0, 3.6, 1.7, 2.0, 0.9, 0.10]

    for i, (line, char_delay) in enumerate(lines):
        for char in line:
            print(char, end="")
            sys.stdout.flush()
            sleep(char_delay)
        sleep(delays[i])
        print("")

print_lyrics()