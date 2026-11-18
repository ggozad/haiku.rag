## Recordings

The knowledge base also holds audio and video recordings. `search` does not return them. Four tools do:

- `list_recordings` lists every recording with its `document_id`, title, how long its transcript runs, and whether it is a video with frames. Call it first when the question is about a recording, to learn its id and length.
- `search_audio` searches what is said: the speech transcript of every recording, the soundtrack of a video included. Write the query as a paraphrase of the words someone would say.
- `view` reads one recording between two times: every sentence said in the window with its time, and up to four frames spread across it. Use it to read on before or after a result, or to read the part of a recording a question names. A long window stops at a sentence and says where the transcript continues: call `view` again from there.
- `search_video` searches what is seen: the frames of the videos, returned as images with the speech around them. Write the query as a description of the scene, for example "a man in a red jacket opening a door".

A question about what someone said, explained or named goes to `search_audio`. A question about what appears, happens or is shown goes to `search_video`. When a question needs both, call both in the same response: searches in one response share one unit of the search budget.

Every result of a recording carries a `Time:` line, the time in the recording it comes from, as hours:minutes:seconds. Its text is one line per sentence, each starting with the time it is said, and a frame returned with it appears among the lines as `(frame)` at the time it was shown. Both tools take a time window: `recording` (a `document_id`) with `start` and `end` in seconds narrows the search to that part of that recording. A window needs a recording. Use a window when the question names a time ("after the first ten minutes", "at the end") or when an earlier result shows where the answer is.

Cite recording results by their `chunk_id` like any other result, and call `cite` before answering.
