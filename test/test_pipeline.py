import unittest
import pandas as pd
from util import preprocess_data

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        """
        Test hàm preprocess_data.
        """
        # Dữ liệu giả lập
        raw_data = {
            "name": ["Alice", "Bob", None, "David"],
            "age": [25, 30, 22, None],
            "city": ["New York", "Los Angeles", "Chicago", "Houston"]
        }
        dataframe = pd.DataFrame(raw_data)

        # Gọi hàm xử lý dữ liệu
        result = preprocess_data(dataframe, ["name", "age"])

        # Kết quả mong đợi
        expected_data = {
            "name": ["Alice", "Bob"],
            "age": [25.0, 30.0]  # Chuyển kiểu `age` thành float để khớp với kết quả trả về
        }
        expected_df = pd.DataFrame(expected_data)

        # So sánh kết quả
        pd.testing.assert_frame_equal(result, expected_df, check_dtype=True)

if __name__ == "__main__":
    unittest.main()
