# Local Script Notes

For the full project handoff, read the workspace root file first:
- `C:\Users\user\Desktop\大三下\EY\AGENTS.md`

This local note only captures the script-folder essentials.

## Main Order

1. `01_build_structured_json.py`
   - Input: `../../../../07_CleanTree精修區/法規資料_md_clean`
   - Output: `../data/json` and `../data/summary`
   - Requires accepted clean md / tree / tables under `07_CleanTree精修區`

2. `02_embed_structured_modes.py`
   - Input: `../data/summary`
   - Output: `../data/embeddings/embedding_bge_m3_<mode>`
   - Main modes: `all_nodes`, `leaf`, `table`, `hybrid`

3. `03_embed_fixed_800_200_baseline.py`
   - Independent baseline
   - Input: `../../../../04_Markdown精修區/法規資料_md`
   - Output: `../data/embeddings/embedding_bge_m3_800200`

4. `04_batch_rag_answers.py`
   - Input: question Excel / CSV plus embeddings
   - Output: answer CSV
   - Requires local Ollama if generating LLM answers

5. `..\app.py`
   - Interactive Streamlit UI

## Important Runtime Note

- In `04_batch_rag_answers.py` and `..\app.py`, `hybrid` mode is assembled at runtime by combining:
  - `leaf` or `all_nodes`
  - `table`
- Do not assume the app is reading the saved `embedding_bge_m3_hybrid` folder directly.

## Do Not

- Do not run all scripts blindly.
- Do not regenerate embeddings before confirming the target mode.
- Do not treat table-audit artifacts as mandatory for the main pipeline.
