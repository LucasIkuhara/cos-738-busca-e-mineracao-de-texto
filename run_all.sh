#!/bin/sh
cd src
python3 query_processor.py &&
python3 word_frequency.py &&
python3 indexer.py &&
python3 search.py
