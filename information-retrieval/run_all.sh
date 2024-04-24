#!/bin/sh
cd information-retrieval
python3 query_processor.py &&
python3 word_frequency.py &&
python3 indexer.py &&
python3 search.py
