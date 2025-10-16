# DTMF Finder Telegram Bot

A Telegram bot that analyzes audio signals to detect telephone keypad button presses using **DTMF (Dual-Tone Multi-Frequency)** signal processing. This project was developed as a term project for a Systems and Signals course.

## Overview

DTMF is the signal generated when you press buttons on a telephone keypad. Each button produces a unique combination of two frequencies - one from a low-frequency group and one from a high-frequency group. This bot analyzes audio recordings to identify which buttons were pressed.

### DTMF Frequency Table

| Button | Low Freq (Hz) | High Freq (Hz) |
|--------|---------------|----------------|
| 1      | 697           | 1209           |
| 2      | 697           | 1336           |
| 3      | 697           | 1477           |
| 4      | 770           | 1209           |
| 5      | 770           | 1336           |
| 6      | 770           | 1477           |
| 7      | 852           | 1209           |
| 8      | 852           | 1336           |
| 9      | 852           | 1477           |
| 0      | 941           | 1336           |
| *      | 941           | 1209           |
| #      | 941           | 1477           |
| A      | 697           | 1633           |
| B      | 770           | 1633           |
| C      | 852           | 1633           |
| D      | 941           | 1633           |

## Features

- Accepts audio messages, voice notes, and WAV files
- Generates spectrograms (before and after filtering)
- Detects DTMF tones from audio recordings
- Configurable detection parameters
- Supports standard keypad (0-9, *, #) and extended keys (A-D)

## How It Works

![Bot Demo](pics/Screenshot%202025-10-16%20193433.png)

### Signal Processing Pipeline

1. **Audio Loading & Preprocessing**
   - Accepts audio in various formats (voice messages, audio files, WAV)
   - Converts stereo to mono if necessary
   - Loads audio using librosa

2. **STFT (Short-Time Fourier Transform)**
   - Transforms time-domain signal to frequency-domain
   - Configurable FFT window size (4410 samples)
   - Hop length: 2205 samples
   - Window function: Hann window

3. **Frequency Amplification**
   - Amplifies DTMF frequency bins by 10x
   - Target frequencies: 697, 770, 852, 941, 1209, 1336, 1477, 1633 Hz
   - Improves signal-to-noise ratio for detection

4. **Threshold Filtering**
   - Calculates 96th percentile of magnitude in DTMF range (691-1640 Hz)
   - Filters out frequencies below threshold
   - Reduces noise and irrelevant frequencies

5. **Mask-Based Key Detection**
   - Creates frequency masks for each button
   - Each mask highlights the two frequencies of a specific key
   - Applies masks to each time slice
   - Computes dot product to find strongest match

6. **Post-Processing**
   - Uses median-based threshold (10x median) to filter noise
   - Removes consecutive duplicate detections
   - Outputs final key sequence

![Detection Results](pics/Screenshot%202025-10-16%20193521.png)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DTMF_finder
```

2. **Install dependencies**
```bash
pip install telebot requests librosa matplotlib numpy python-dotenv
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```
API_TOKEN=your_telegram_bot_token_here
```

Get your bot token from [@BotFather](https://t.me/botfather) on Telegram.

4. **Run the bot**
```bash
python main.py
```

## Usage

### Commands

- `/start` - Get welcome message
- `/help` - Get usage instructions
- `/set <parameter> <value>` - Adjust detection parameters
- `/getsettings` - View current parameter values

### Basic Usage

1. Start a chat with your bot on Telegram
2. Send an audio message, voice note, or WAV file containing DTMF tones
3. The bot will:
   - Send the original spectrogram
   - Send the filtered spectrogram
   - Send preliminary detected keys
   - Send final detected key sequence

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_FFT` | 4410 | FFT window size |
| `HOP_LEN` | 2205 | Hop length for STFT |
| `WIN_LEN` | 4410 | Window length |
| `MAGNITUDE_AMP_MULTIPLIER` | 10.0 | DTMF frequency amplification factor |
| `THRESHOLD_PERCENTILE` | 96.0 | Percentile used for threshold filtering |
| `MASKS_MEDIAN_MULTIPLIER` | 10.0 | Median multiplier for key extraction |
| `PLT_YLIM_LOWER` | 0 | Lower frequency limit for plots (Hz) |
| `PLT_YLIM_UPPER` | 16384 | Upper frequency limit for plots (Hz) |
| `FIG_SIZE_X` | 10 | Spectrogram width |
| `FIG_SIZE_Y` | 4 | Spectrogram height |

They are needed to adjust for different audio signals and to make the bot more accurate.

