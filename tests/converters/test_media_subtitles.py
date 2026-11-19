import shutil

import pytest

from haiku.rag.converters import media
from haiku.rag.converters.media import MediaConverter, Word, parse_cues
from tests.converters.test_media_converter import (
    DATA,
    _config,
    _segments,
    transcription_server,  # noqa: F401
)
from tests.platform import needs_ffmpeg

VTT = """﻿WEBVTT
Kind: captions

NOTE a comment
spanning lines --> not a cue

STYLE
::cue { color: lime }

intro
00:00.500 --> 00:01.250 align:start position:10%
<v Ann>Hello</v>

00:00:01.300 --> 00:00:02.000
<i>world</i> &amp; all.

00:00:02.100 --> 00:00:03.000
Two lines
of text

00:00:03.100 --> 00:00:04.000
<i></i>
"""

SRT = """1
00:00:00,500 --> 00:00:01,250
Hello

2
00:00:01,300 --> 00:00:02,000
world.
"""


def test_vtt_cues_are_timed_words_without_markup():
    assert parse_cues(VTT) == [
        Word(0.5, 1.25, " Hello"),
        Word(1.3, 2.0, " world & all."),
        Word(2.1, 3.0, " Two lines of text"),
    ]


def test_srt_cues_are_timed_words():
    assert parse_cues(SRT.replace("\n", "\r\n")) == [
        Word(0.5, 1.25, " Hello"),
        Word(1.3, 2.0, " world."),
    ]


def test_hours_count_in_a_cue_time():
    assert parse_cues("WEBVTT\n\n01:02:03.500 --> 01:02:04.000\nlate\n") == [
        Word(3723.5, 3724.0, " late")
    ]


@needs_ffmpeg
@pytest.mark.parametrize(("suffix", "subtitles"), [(".vtt", VTT), (".srt", SRT)])
async def test_subtitles_beside_a_recording_are_its_transcript(
    tmp_path,
    transcription_server,  # noqa: F811
    suffix,
    subtitles,
):
    recording = tmp_path / "talk.mp3"
    shutil.copy(DATA / "speech.mp3", recording)
    (tmp_path / "talk").with_suffix(suffix).write_text(subtitles, encoding="utf-8")
    seen = transcription_server(lambda request: _segments((0.0, 1.0, "never")))

    doc = await MediaConverter(_config()).convert_file(recording)

    assert seen == []
    timed = [(t.source[0].start_time, t.source[0].end_time, t.text) for t in doc.texts]
    assert timed[0][:2] == (0.5, 2.0)
    assert timed[0][2].startswith("Hello world")


@needs_ffmpeg
async def test_empty_subtitles_give_no_transcript(
    tmp_path,
    transcription_server,  # noqa: F811
):
    recording = tmp_path / "talk.mp3"
    shutil.copy(DATA / "speech.mp3", recording)
    (tmp_path / "talk.vtt").write_text("WEBVTT\n", encoding="utf-8")
    seen = transcription_server(lambda request: _segments((0.0, 1.0, "never")))

    doc = await MediaConverter(_config()).convert_file(recording)

    assert seen == []
    assert not doc.texts


def test_a_recording_without_subtitles_has_no_sidecar(tmp_path):
    (tmp_path / "talk.mp3").touch()
    (tmp_path / "other.vtt").touch()

    assert media.subtitles(tmp_path / "talk.mp3") is None


def test_a_sentence_ends_at_its_latest_cue_end():
    overlapping = (
        "WEBVTT\n\n00:00.000 --> 00:05.000\nLong cue\n\n"
        "00:01.000 --> 00:02.000\nshort one.\n"
    )
    [sentence] = media.sentences(
        [media.Utterance(0.0, 5.0, "", tuple(parse_cues(overlapping)))]
    )

    assert (sentence.start, sentence.end) == (0.0, 5.0)
