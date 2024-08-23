import pandas as pd
from typing import Literal

class importData:
    """A class for loading the dataset.

    Args:
        file_name (str): name of dataset file
        file_path (str): path containing dataset
        data_type (str): type of dataset, choose from csv, xlsx
    """
    def __init__(self, file_name: str, file_path: str = "./", data_type: Literal['csv', 'xlsx'] = "csv") -> None:
        self.file_name = file_name
        self.file_path = file_path
        self.data_type = data_type

    def get_file(self) -> pd.DataFrame:
        """Get dataset as a pandas dataframe

        Returns:
            data (pd.DataFrame): loaded dataset
        """
        if self.data_type == "csv":
            data = pd.read_csv(self.file_path + self.file_name, sep=';', encoding='ISO-8859-1')
        if self.data_type == "xlsx":
            data = pd.read_excel(self.file_path + self.file_name)
        return data