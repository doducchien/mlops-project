import pandas as pd

def preprocess_data(dataframe, columns_to_keep):
    """
    Hàm xử lý dữ liệu: Lọc các cột cần thiết và xóa các giá trị null.
    
    Args:
        dataframe (pd.DataFrame): DataFrame đầu vào.
        columns_to_keep (list): Danh sách các cột cần giữ lại.

    Returns:
        pd.DataFrame: DataFrame đã được xử lý.
    """
    # Lọc các cột cần thiết
    filtered_df = dataframe[columns_to_keep]
    # Loại bỏ các hàng có giá trị null
    cleaned_df = filtered_df.dropna()
    # Ép kiểu dữ liệu về dạng nguyên gốc (nếu cần thiết)
    cleaned_df = cleaned_df.astype(dataframe[columns_to_keep].dtypes)
    return cleaned_df
