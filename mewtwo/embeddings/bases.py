from typing import Optional
from dataclasses import dataclass


@dataclass
class BasePair:
    base_1: Optional[str]
    base_2: Optional[str]
    h_bonded: bool

    def __repr__(self):
        repr_base_1 = self.base_1
        repr_base_2 = self.base_2

        if self.base_1 is None:
            repr_base_1 = ' '

        if self.base_2 is None:
            repr_base_2 = ' '

        if self.h_bonded:
            return f'{repr_base_1}-{repr_base_2}'
        else:
            return f'{repr_base_1}.{repr_base_2}'

    def __eq__(self, other):

        if self.base_1 != other.base_1:
            return False

        if self.base_2 != other.base_2:
            return False

        if self.h_bonded != other.h_bonded:
            return False

        return True

    def __hash__(self):
        return self.base_1, self.base_2, self.h_bonded


