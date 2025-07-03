import os
import builtins
from unittest.mock import patch, mock_open
import pytest
from models.api_model import send_generation_request

# --- Pytest fixture to mock the requests.post call ---
@pytest.fixture
def mock_requests_post():
    """
    This fixture mocks the 'requests.post' method used in the API call inside send_generation_request.
    Instead of making an actual HTTP request, it returns a pre-defined MockResponse object.
    This allows tests to run offline, deterministically, and quickly without relying on external services.
    """
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.content = b"fake_image_data"  # Simulated image bytes from API
            self.headers = {"seed": "12345", "finish-reason": "SUCCESS"}

        @property
        def ok(self):
            return True  # Simulates a successful HTTP response

    # Patch 'requests.post' globally in the module during the test
    with patch("requests.post", return_value=MockResponse()) as mock_post:
        yield mock_post  # Provide the mocked post method to any test that wants it


# --- Pytest fixture to mock the built-in open function ---
@pytest.fixture
def mock_file_open():
    """
    This fixture mocks Python's built-in 'open' function used for both reading style images and writing output files.
    By using 'mock_open', we simulate file operations in memory without touching the disk.
    This helps avoid side effects during tests and enables verification of file handling logic.
    """
    m_open = mock_open(read_data=b"fake_style_image_data")  # Fake data for style image reading

    # Patch 'open' in the builtins module used by api_model code
    with patch.object(builtins, "open", m_open):
        yield m_open  # Provide the mocked open to tests for inspection


# --- Fixture to call the send_generation_request function with mocks applied ---
@pytest.fixture
def generated_image_path(mock_requests_post, mock_file_open):
    """
    This fixture runs the actual function under test using the above mocks.
    It returns the generated image path string output by the function.
    By using this fixture, we avoid duplicating setup code across multiple test cases.
    """
    params = {
        "prompt": "a cat in a hat",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "model": "sd3.5",
        "seed": 1,
        "image": "dummy_path.jpg",  # Path passed to function, but open is mocked
    }
    user_id = "testuser"
    iteration = 1
    session_num = 1
    host = "https://fake.api"

    # Call the function and capture the returned file path
    path = send_generation_request(host, params, user_id, iteration, session_num)
    return path



# --- Actual tests ---

def test_return_type(generated_image_path):
    # Check that the returned value is a string (the path to saved image)
    assert isinstance(generated_image_path, str)

def test_file_extension(generated_image_path):
    # Ensure the returned filename ends with '.png' as expected
    assert generated_image_path.endswith(".png")

def test_file_exists(mock_file_open, generated_image_path):
    # Since file writing is mocked, the file does not exist on disk.
    # Instead, verify that open was called with the correct filename and 'wb' (write binary) mode,
    # which means the function attempted to write the image file.
    mock_file_open.assert_any_call(generated_image_path, "wb")

def test_filename_format(generated_image_path):
    # Confirm the filename contains the expected pattern based on user_id, session, and iteration
    expected_filename = "testuser_session1_iter1.png"
    assert expected_filename in generated_image_path
