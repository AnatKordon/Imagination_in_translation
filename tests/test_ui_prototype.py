import pytest
from ui import ui

def test_input_is_english():
    fname = pathlib.Path(__file__)
    q = QuestionnaireAnalysis(fname)
    assert fname == q.data_fname

# Unit test to verify that prompt text is in English letters only

def is_english(text):
    # Allow English letters, digits, spaces, and basic punctuation
    return re.fullmatch(r"[A-Za-z0-9\s.,!?'\"]+", text) is not None

@pytest.mark.unit
@pytest.mark.parametrize("input_text, expected", [
    ("This is an English sentence.", True),
    ("תיאור בעברית בלבד", False),
    ("Description123 with numbers", True),
    ("טקסט משולב English עברית", False),
])
def test_prompt_is_english(input_text, expected):
    assert is_english(input_text) == expected
