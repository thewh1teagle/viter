"""Real-corpus test on the 20-utterance LJSpeech sample. Skipped unless the data is present."""

import viter


def test_train_and_align_lj20(lj20, tmp_path):
    corpus, dict_path = lj20
    model = viter.train(corpus, dict=dict_path, no_lda=True, cpu=True)
    assert model.phones

    wav = sorted(corpus.glob("**/*.wav"))[0]
    text = wav.with_suffix(".txt").read_text().strip()
    a = model.align(wav, text)
    assert [w.label for w in a.words] == text.split()
    assert a.duration > 0
    assert a.phones

    tg = tmp_path / "out.TextGrid"
    a.to_textgrid(tg)
    assert tg.read_text().startswith('File type = "ooTextFile"')
