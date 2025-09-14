# Expression Visualization Summary

## Key Findings

### 1. Expression Embeddings ARE Varying
From the dataset debug output, we confirmed that expression embeddings are being extracted per-frame correctly:
- Constant across frames: **False**
- Mean frame-to-frame difference: **0.008542**
- Max frame-to-frame difference: **0.021958**

This shows the expressions DO vary across frames, contrary to what the wandb visualizations suggested.

### 2. Created Visualization Tools

#### `/media/2TB/VASA-1-hack/nemo/visualize_expressions_and_warps.py`
Main visualization script with two functions:
- `create_expression_candles()` - Creates candlestick visualization showing statistics per frame
- `visualize_warping_field()` - Visualizes warping/deformation fields

#### `/media/2TB/VASA-1-hack/nemo/test_expression_candles.py`
Test script that generates synthetic data to validate visualization:
- Creates expression data with temporal variations
- Generates both varying and constant expressions for comparison
- Creates synthetic warping fields

#### `/media/2TB/VASA-1-hack/nemo/visualize_from_checkpoint.py`
Script to visualize expressions from training checkpoints (if stored)

### 3. Generated Visualizations

Successfully created in `nemo/visualization_output/`:
- `synthetic_expression_candles_variation.png` - Shows varying expressions
- `synthetic_expression_candles_constant.png` - Shows constant expressions (for comparison)
- `synthetic_warp_field.png` - Shows warping field visualization

### 4. Expression Candle Visualization Format

The candle visualization shows:
1. **Top subplot**: Candlestick view
   - Box: 25th to 75th percentile of expression values per frame
   - Whiskers: Min to max values
   - Red line: Mean value

2. **Bottom subplot**: Heatmap
   - All 128 expression dimensions over time
   - Color represents expression values

### 5. Why Wandb Shows No Variation

The issue is likely in the wandb logging code in `vasa_trainer.py`:
- The visualization uses `visualize_expression.py` which already exists
- It may be logging the wrong tensor or the same frame repeatedly
- The actual data HAS variation as confirmed by our debug output

### 6. Next Steps

To fix wandb visualization:
1. Check what's being passed to `create_expression_candles()` in vasa_trainer.py
2. Ensure `targets['expression_embed']` contains the full temporal sequence
3. Verify the shape is [B, T, 128] not [B, 1, 128]

The expressions ARE being extracted correctly per-frame, the issue is only in visualization/logging.