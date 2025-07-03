import os
from models.api_model import send_generation_request

def test_basic_send(monkeypatch):
    # Monkeypatch requests.post to avoid real API calls
    class MockResponse:
        status_code = 200
        content = b"fake_image_data"
        headers = {"seed": "12345", "finish-reason": "SUCCESS"}

        def ok(self):
            return True

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("requests.post", mock_post)

    # Minimal test call
    params = {
        "prompt": "a cat in a hat",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "model": "sd3.5",
        "seed": 1,
        "session": 1
    }

    image_path = send_generation_request(
        host="https://fake.api",
        params=params,
        user_id="testuser",
        iteration=1,
        session_num=1
    )

    # Assert the returned path is a string and ends with .png
    assert isinstance(image_path, str)
    assert image_path.endswith(".png")

    # Assert the file actually exists on disk
    assert os.path.exists(image_path), "Generated image file was not saved."

    # check the file size or content is what was mocked
    with open(image_path, "rb") as f:
        content = f.read()
    assert content == b"fake_image_data", "Saved image content does not match expected data."

    # Check the filename format
    expected_filename = f"testuser_session1_iter1.png"
    assert expected_filename in image_path, "Filename does not match expected format."
