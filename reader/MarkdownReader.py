from pathlib import Path
from typing import List
from .base import Reader

class MarkdownReader(Reader):
    def __init__(self):
        pass

    def read_folder(self, folder_path) -> List[str]:
        return [file for file in Path(rf'{folder_path}').rglob('*.md')]