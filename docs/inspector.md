# Database Inspector

Interactive TUI for browsing your LanceDB database.

## Installation

```bash
# For haiku.rag-slim
pip install 'haiku.rag-slim[inspector]'

# Already included in haiku.rag
pip install haiku.rag
```

## Usage

```bash
haiku-rag inspect
haiku-rag inspect --db /path/to/database.lancedb
```

## Interface

![Inspector with search](img/inspector-search.svg)

Three panels display your data:

- **Documents** (left) - All documents in the database
- **Chunks** (top right) - Chunks for the selected document
- **Detail View** (bottom right) - Full content and metadata

## Navigation

**Keyboard:**

- `Tab` - Cycle between panels
- `↑` / `↓` - Navigate lists
- `/` - Open search modal
- `c` - Open context expansion modal (when viewing a chunk)
- `v` - Open visual grounding modal (when viewing a chunk)
- `m` - Browse multimodal image assets (`mm_assets`) for the selected document
- `d` - Show whole-document index overlay (all chunk boxes + all image asset boxes)
- `q` - Quit

**Mouse:** Click to select, scroll to view content

## Search

Press `/` to open the full-screen search modal:

- Enter your query and press `Enter` to search
- **Left panel**: Search results with relevance scores `[0.95] content preview`
- **Right panel**: Full chunk content and metadata
- Use `↑` / `↓` to navigate results - detail view updates in real-time
- Press `Enter` on a result to close search and navigate to that document/chunk
- Press `Esc` to close search without selecting

Search uses hybrid (vector + full-text) search across all chunks. Content is rendered as markdown with syntax highlighting.

## Context Expansion

Press `c` while viewing a chunk to open the context expansion modal:

- Shows the expanded context that would be provided to the QA agent
- Type-aware expansion: tables, code blocks, and lists expand to their complete structures
- Text content expands based on `search.context_radius` setting
- Includes metadata like source document, content type, and relevance score
- Press `Esc` to close the modal

## Visual Grounding

Press `v` while viewing a chunk to open the visual grounding modal:

- Shows page images from the source document with the chunk's location highlighted in yellow/orange
- Use `←` / `→` arrow keys to navigate between pages (when chunk spans multiple pages)
- Press `o` to open the current page image in your OS image viewer (recommended for terminals that don't render images inline)
- Press `Esc` to close the modal

!!! note
    Visual and export images are written to a temporary directory and **deleted when you close the modal**.
    If you want to keep an image, press `o` to open it and save/copy it from your OS viewer before closing.

## Multimodal Assets (mm_assets)

Press `m` to open the multimodal assets modal for the currently selected document:

- Lists all `mm_assets` rows (one per indexed picture crop)
- Press `p` to toggle between `crop` and `page` modes
- Press `o` to export + open the current selection in your OS image viewer
  (exports are stored in a temporary directory and deleted when you close the modal)

## Whole-Document Index Overlay

Press `d` to open the whole-document overlay:

- Exports per-page PNGs with **all chunk bounding boxes** (yellow/orange) and **all mm_asset bounding boxes** (cyan/blue)
- Use `←` / `→` to navigate pages
- Press `o` to open the current page PNG in your OS viewer
  (exports are stored in a temporary directory and deleted when you close the modal)

!!! note
    Visual grounding requires documents with a stored DoclingDocument that includes page images.
    Text-only documents or documents imported without DoclingDocument won't have visual grounding available.
