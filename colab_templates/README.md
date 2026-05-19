# Colab templates (AI Matte)

Desk **Send to Colab** uploads `ai_matte_cellcode_template.py` to Drive as `VDA_Jobs/code/{job_id}_cellcode.py`.

Keep this folder in sync with `google_desk_app/colab_templates/` when you change Colab behavior.

| File | Role |
|------|------|
| `ai_matte_cellcode_template.py` | Cell 1–3 functions (`setup_ai_matte_cell1`, …) |
| `ai_matte_notebook_minimal_template.py` | Minimal 3-cell notebook skeleton |

**Important:** `CODE_FILE_ID` in the notebook must match the **same job’s** `_cellcode.py` on Drive. A VDA/DVD cellcode ID causes `ImportError: setup_ai_matte_cell1`.
