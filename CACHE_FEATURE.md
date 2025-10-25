# Comparison Cache Feature

## Overview
The Compare Models tab now includes intelligent caching to avoid re-running the same 12 model combinations for files you've already analyzed.

## How It Works

### Cache Key Generation
- When you upload a file, the system creates a unique cache key based on:
  - **File content hash** (SHA-256): Ensures the exact same file
  - **Analysis depth**: Quick Scan / Standard Analysis / Deep Dive
  - **Custom rubric**: None / Sales Outbound / Banking Support / Technical Support

### Cache Behavior

1. **First Run**: 
   - Upload a file and run comparison
   - Results are stored in memory for the session
   - Cache indicator shows "🔄 Not cached"

2. **Subsequent Runs**:
   - Upload the same file with the same settings
   - Cache indicator shows "✅ Cached"
   - Green banner: "💾 Using cached results from previous analysis"
   - No API calls are made - instant results!

3. **Different Settings**:
   - Same file but different depth/rubric = new cache entry
   - Each unique combination is cached separately

## Cache Management

### View Cache Status
- **Main area**: Shows "✅ Cached" or "🔄 Not cached" next to combination count
- **Sidebar**: Shows total number of cached comparisons

### Clear Cache
Two ways to clear:
1. **"🗑️ Clear Cache"** button (next to download button)
2. **"Clear All Cache"** button in sidebar

### Cache Persistence
- Cache lives for the duration of your Streamlit session
- Closing the browser/tab clears the cache
- Refreshing the page clears the cache
- Cache is stored in `st.session_state.comparison_cache`

## Benefits

### Time Savings
- **First run**: ~2-5 minutes for 12 combinations (depending on file length)
- **Cached run**: Instant display of results
- **Perfect for**: Testing different visualizations or re-examining results

### Cost Savings
- No repeated API calls to:
  - STT engines (Whisper, Gladia, Deepgram, AssemblyAI)
  - Analysis models (Gemini, GPT-4, Claude)
- Significant cost reduction when iterating on analysis

### Use Cases

1. **Iterative Analysis**
   - Upload file, run comparison, view results
   - Realize you want to see the data differently
   - Results load instantly without re-running

2. **Presentation Mode**
   - Run analysis before a meeting
   - Demo the results multiple times without waiting
   - Cache persists through the entire session

3. **A/B Comparison**
   - Run "Quick Scan" for file X → cached
   - Run "Standard Analysis" for file X → cached separately
   - Switch between both instantly to compare depth impact

## Technical Details

### Cache Structure
```python
st.session_state.comparison_cache = {
    "cache_key_1": {
        "file": "recording.wav",
        "depth": "Standard Analysis",
        "rubric": None,
        "data": [12 comparison results]
    },
    "cache_key_2": {
        "file": "recording.wav",
        "depth": "Deep Dive",
        "rubric": "Sales Outbound",
        "data": [12 comparison results]
    }
}
```

### Cache Key Example
```python
file_hash = hashlib.sha256(file_content).hexdigest()
cache_key = f"{file_hash}_Standard Analysis_None"
# Result: "a3f7b9e2...8c4d_Standard Analysis_None"
```

## Limitations

1. **Session-only**: Cache doesn't persist across browser sessions
2. **Memory-based**: Large number of cached results consume RAM
3. **No disk storage**: Results not saved to disk (by design for privacy)

## Future Enhancements (Potential)

- [ ] Persistent cache using local storage
- [ ] Cache expiration (time-based)
- [ ] Cache size limits with LRU eviction
- [ ] Export/import cache for session recovery
- [ ] Cache analytics (hit rate, savings metrics)

## Example Workflow

```
1. Upload: sample_call.wav
   Settings: Standard Analysis, No rubric
   Action: Click "Run Full Model Comparison"
   Result: 12 combinations analyzed in 3 minutes
   Status: "💾 Results cached for future use!"

2. Same file, same settings
   Action: Just upload the file again
   Result: Instant display - "💾 Using cached results"
   Time saved: 3 minutes
   
3. Same file, different depth
   Settings: Deep Dive, No rubric  
   Action: Click "Run Full Model Comparison"
   Result: New analysis runs (different cache key)
   Status: Both caches now available

4. Later in session
   Switch back to "Standard Analysis"
   Result: Instantly loads first cache
   No re-analysis needed!
```

## Tips

- Always check the cache indicator before running
- Use "Quick Scan" for faster initial caching
- Clear cache if you suspect stale results
- Cache is your friend for demos and presentations!
