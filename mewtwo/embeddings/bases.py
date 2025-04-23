from typing import Optional
from dataclasses import dataclass
from enum import Flag


class Base(Flag):
    A = 1
    C = 2
    G = 4
    T = 8
    U = 16

    def __repr__(self):
        return self.name


DNA_BASES = Base.A | Base.T | Base.G | Base.C
RNA_BASES = Base.A | Base.U | Base.G | Base.C

WATSON_CRICK_PAIRS = [Base.A | Base.T,
                      Base.A | Base.U,
                      Base.C | Base.G]

WOBBLE_PAIRS = [Base.G | Base.U]


@dataclass
class BasePair:
    base_1: Optional[Base]
    base_2: Optional[Base]
    h_bonded: bool

    def __repr__(self):

        if self.base_1 is None:
            repr_base_1 = ' '
        else:
            repr_base_1 = self.base_1.name

        if self.base_2 is None:
            repr_base_2 = ' '
        else:
            repr_base_2 = self.base_2.name

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
        return hash((self.base_1, self.base_2, self.h_bonded))

    def is_watson_crick(self):
        if self.base_1 and self.base_2 and self.base_1 | self.base_2 in WATSON_CRICK_PAIRS:
            return True

        return False

    def is_wobble(self):
        if self.base_1 and self.base_2 and self.base_1 | self.base_2 in WOBBLE_PAIRS:
            return True

        return False



