# Word Frequency & Word Cloud Integration - Implementation Summary

## Overview

Word frequency tracking is integrated into the academic ingestion pipeline to support
word cloud visualisation on the dashboard. This tracks the most common words across all
ingested documents for visual analytics.

## Components Implemented

### 1. Database Storage (scripts/ingest/cache_db.py)

**Table: `word_frequency`**
- Schema:
  - `word` (TEXT PRIMARY KEY): The word
  - `frequency` (INTEGER): Total occurrence count across all documents
  - `doc_count` (INTEGER): Number of unique documents containing the word
  - `last_updated` (TIMESTAMP): When stats were last updated

- Indexes:
  - `idx_word_frequency`: Optimises sorting by frequency DESC
  - `idx_word_doc_count`: Optimises filtering by document count

**Associated Methods:**
- `put_word_frequencies(word_freqs, doc_count=None)`: Store accumulated frequencies
- `get_top_words(limit=100, min_frequency=1)`: Retrieve top N words by frequency
- `get_word_frequency_stats()`: Get overall statistics
- `clear_word_frequencies()`: Reset all data

### 2. Word Frequency Extraction (scripts/ingest/word_frequency.py)

**WordFrequencyExtractor Class**
- Tokenises text using TextPreprocessor (LOWERCASE strategy)
- Filters stopwords automatically
- Returns word -> frequency mapping
- Min word length: 2 characters (configurable)

**Key Features:**
- Consistent with domain terminology extraction (same stopword filtering)
- Preserves word identity (no stemming/lemmatisation)
- Document frequency tracking (for filtering common but less-informative words)

**Usage:**
```python
from scripts.ingest.word_frequency import WordFrequencyExtractor

extractor = WordFrequencyExtractor(min_word_length=2)
word_freqs = extractor.extract_frequencies(text)
# Returns: {'leadership': 45, 'workplace': 28, ...}
```

### 3. Academic Ingestion Integration (scripts/ingest/ingest_academic.py)

**Ingestion Pipeline Flow:**
1. Initialise `WordFrequencyExtractor` at pipeline start
2. Extract frequencies from each primary document's text
3. Extract frequencies from each reference artifact
4. Accumulate all frequencies in `Counter` object
5. Store final accumulated frequencies in cache database after processing

**Code Changes:**
- Line 18: Import `WordFrequencyExtractor`
- Line 632-633: Initialise extractor and accumulator
- Line 653-655: Extract frequencies from primary documents
- Line 772-774: Extract frequencies from reference texts
- Line 854-876: Store, report, and log word frequency statistics

**Reporting:**
After ingestion completes, logs include:
- Total unique words tracked
- Total word frequency
- Average frequency per word
- Top 20 words for word cloud

### 4. Dashboard Integration (scripts/ui/word_cloud_provider.py)

**Public API:**
- `get_word_cloud_data(limit=100, min_frequency=1, format='array')`
  - Formats: array, object, weighted
  - Returns data suitable for D3.js / Plotly visualisation
  
- `get_word_cloud_stats()`
  - Returns: total_unique_words, total_frequency, avg/max/min frequency

**Format Examples:**

Array format (default):
```json
[{"word": "leadership", "frequency": 2194, "doc_count": 45}, ...]
```

Object format:
```json
{"leadership": {"frequency": 2194, "doc_count": 45}, ...}
```

Weighted format (for sizing):
```json
[{"word": "leadership", "size": 1.0, "frequency": 2194}, ...]
```

### 5. Testing (tests/test_word_frequency.py)

**Test Coverage:**
- Extraction: Basic frequencies, stopword filtering, min length, doc counts
- Cache: Storage, retrieval, accumulation, statistics, clearing
- Provider: Array/object/weighted formats, statistics, error handling

## Usage Examples

### During Ingestion

```bash
# Word frequencies are collected automatically
python scripts/ingest/ingest_academic.py --papers ~/thesis.pdf --domain "leadership"
# Output: "Top 20 words for word cloud:"
# 1. leadership        freq= 2194, doc_count= 45
# 2. indigenous        freq= 1028, doc_count= 38
# ...
```

### Querying for Dashboard

```python
from scripts.ui.word_cloud_provider import get_word_cloud_data

# Array format for dropdown table
data = get_word_cloud_data(limit=50, format='array')

# Weighted format for word cloud
weighted_data = get_word_cloud_data(limit=100, format='weighted')
# Use data[*].word and data[*].size for visualisation

# Statistics
stats = get_word_cloud_stats()
# {
#   'total_unique_words': 3847,
#   'total_frequency': 45230,
#   'avg_frequency': 11.76,
#   'max_frequency': 2194,
#   'min_frequency': 1
# }
```

## Data Flow

```
Ingestion:
    1. Load document text
    2. Extract word frequencies (WordFrequencyExtractor)
    3. Accumulate in Counter
    4. Load reference artifacts
    5. Extract word frequencies from each
    6. Accumulate in Counter
    7. Store accumulated frequencies in cache_db.word_frequency table

Dashboard Query:
    1. Call get_word_cloud_data(format='weighted', limit=100)
    2. Retrieve from cache_db.word_frequency (sorted by frequency DESC)
    3. Normalise frequencies to 0-1 scale for sizing
    4. Return formatted data to dashboard
    5. Render word cloud 
```

## Performance Characteristics

- **Ingestion**: ~50-100ms per document (tokenisation + counting)
- **Storage**: O(V) space where V = vocabulary size (~4000-10000 words)
- **Queries**: <10ms for top N words (indexed by frequency)
- **Database Size**: ~50-100KB for typical academic document corpus

## Configuration

**Word Length**: Controlled in WordFrequencyExtractor
```python
extractor = WordFrequencyExtractor(min_word_length=3)  # Skip 2-letter words
```

**Stopwords**: Shared with domain terminology (in `text_preprocessing.py`)
- NLTK English stopwords
- Domain-specific filters (citations, PDF artifacts, etc.)

**Frequency Threshold**: In dashboard queries
```python
get_word_cloud_data(min_frequency=5)  # Exclude rare words
```

## Optionality

- ✅ Word frequencies are optional
- ✅ Graceful fallback if word frequency table unavailable

## TODO: uture Enhancements

- [ ] N-gram frequencies (bigrams, trigrams along the lines of domain terms)
- [ ] Time-series word frequency (track evolution across documents)
- [ ] Domain-specific word weighting (eg. boost leadership-related words if documents are on leadership)
- [ ] Comparison word clouds (thesis vs references)
- [ ] TF-IDF variant for importance weighting or potentially a domain terms comparison

## Associated Files

1. **scripts/ingest/cache_db.py**
   - Wword_frequency table schema
   - CRUD methods for word frequencies
   - Indexes for performance

2. **scripts/ingest/word_frequency.py**
   - WordFrequencyExtractor class
   - Convenience function for extraction

3. **scripts/ingest/ingest_academic.py**
   - Import WordFrequencyExtractor
   - Initialise extractor and accumulator
   - Collect frequencies during ingestion
   - Store final frequencies at completion
   - Log statistics and top words

4. **scripts/ui/word_cloud_provider.py**
   - get_word_cloud_data() - public API
   - get_word_cloud_stats() - statistics API
   - Support for array/object/weighted formats

5. **tests/test_word_frequency.py**
   - Unit tests
   - Extraction, storage, retrieval, formatting

## Testing

Run tests:
```bash
pytest tests/test_word_frequency.py -v
```
