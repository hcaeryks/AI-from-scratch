from typing import Generator
import csv

class Dataset():
    def __init__(self, path_to_data: str) -> None:
        self.data = []
        with open(path_to_data, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append(row)

    def batch(self, batch_size: int) -> Generator:
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i + batch_size]