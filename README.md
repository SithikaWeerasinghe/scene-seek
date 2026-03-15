# SceneSeek

SceneSeek is a prototype movie identification tool, similar to Shazam but for films.
It identifies a movie from a short uploaded video clip by extracting frames and matching them against a reference database.

## Overview

This project is a small, clean prototype developed for an intern career day demo. It demonstrates the process of:
1. Extracting frames from a video clip.
2. Generating visual features from those frames.
3. Comparing features against a reference movie database.
4. Returning the most likely movie match.

## Project Structure

- `data/reference`: Known movie samples for the reference database.
- `data/test`: Clips used for testing the identification logic.
- `src/preprocess`: Logic for video handling and frame extraction.
- `src/features`: Feature extraction logic from frames.
- `src/matcher`: Logic for matching features against the database.
- `src/app`: Main application flow and entry points.
- `src/utils`: Helper functions and common utilities.
- `outputs`: Directory for saved results, extracted frames, or prediction outputs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python src/app/main.py
```
