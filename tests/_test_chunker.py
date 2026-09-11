"""_test_chunker.py — chunker.split_text / chunk_document"""
from _runner import Runner

r = Runner("chunker")


def t01_thai():
    from chunker import split_text
    chunks = split_text("การเรียนรู้ของเครื่องคือสาขาหนึ่งของปัญญาประดิษฐ์ " * 20, 200, 40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250, f"chunk too long: {len(c)}"


def t02_english():
    from chunker import split_text
    chunks = split_text("Machine learning is a subset of AI. " * 30, 200, 40)
    assert len(chunks) > 1


def t03_parent_child():
    from chunker import chunk_document
    pairs = chunk_document("hello world " * 100)
    assert len(pairs) > 0
    assert "parent_text" in pairs[0] and "child_text" in pairs[0]
    assert len(pairs[0]["child_text"]) <= 420


def t04_newline_split():
    from chunker import split_text
    text = "บทที่ 1\n\nเนื้อหาแรก\n\nบทที่ 2\n\nเนื้อหาสอง"
    chunks = split_text(text, 20, 5)
    assert len(chunks) >= 2


def t42_overlap_never_exceeds_chunk_size():
    from chunker import split_text
    small = "ก" * 70
    large = "ข" * 350
    chunks = split_text(small + "\n\n" + large, 400, 80)
    assert max(map(len, chunks)) <= 400, [len(c) for c in chunks]


def t43_unbroken_token_is_hard_bounded():
    from chunker import split_text
    chunks = split_text("ภาษาไทย " + ("A" * 700), 100, 20)
    assert max(map(len, chunks)) <= 100, [len(c) for c in chunks]


def t44_inline_time_or_ratio_is_preserved():
    from chunker import split_text
    chunks = split_text("12:34\nอัตราส่วนสำคัญคือ 12:34 และเวลา [01:23]", 400, 80)
    text = "\n".join(chunks)
    assert "อัตราส่วนสำคัญคือ 12:34" in text
    assert "[01:23]" not in text


r.test("T01 Thai word boundary", t01_thai)
r.test("T02 English fallback", t02_english)
r.test("T03 parent/child pairs", t03_parent_child)
r.test("T04 newline split priority", t04_newline_split)
r.test("T42 overlap stays within configured size", t42_overlap_never_exceeds_chunk_size)
r.test("T43 unbroken token is hard-bounded", t43_unbroken_token_is_hard_bounded)
r.test("T44 preserve inline time/ratio while cleaning timestamps", t44_inline_time_or_ratio_is_preserved)

if __name__ == "__main__":
    r.exit()
