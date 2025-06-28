from abc import ABC, abstractmethod

from typing import List

class Reader():
    @abstractmethod
    def read_folder(self, folder_path) -> List[str]:
        pass