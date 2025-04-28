from typing import Optional
from dataclasses import dataclass
from enum import Flag


class PairingType(Flag):
    STRUCTURE_BASED = 1
    WATSON_CRICK = 2
    WOBBLE = 4
    WOBBLE_OR_WATSON_CRICK = WOBBLE | WATSON_CRICK


class Base(Flag):
    A = 1
    C = 2
    G = 4
    T = 8
    U = 16
    ZERO_PADDING = 32
    DNA = A | T | G | C
    RNA = A | U | G | C
    PURINES = A | G
    PYRIMIDINES = C | T | U
    TWO_H_BONDS = A | T | U
    THREE_H_BONDS = C | G


BASE_TO_ONEHOT = {Base.A: [1, 0, 0, 0],
                  Base.C: [0, 1, 0, 0],
                  Base.G: [0, 0, 1, 0],
                  Base.T: [0, 0, 0, 1],
                  Base.U: [0, 0, 0, 1],
                  Base.ZERO_PADDING: [0, 0, 0, 0]}


def base_to_vector(base: Base, one_hot: bool = False) -> list[int]:

    if one_hot:
        if base not in BASE_TO_ONEHOT:
            raise ValueError(f"Not a base: {base}")
        else:
            return BASE_TO_ONEHOT[base][:]

    else:
        if base in Base.PURINES:
            element_1 = 1
            element_2 = 0
        elif base in Base.PYRIMIDINES:
            element_1 = 0
            element_2 = 1

        elif base == Base.ZERO_PADDING:
            element_1 = 0
            element_2 = 0
        else:
            raise ValueError(f"Unknown base: {base}")

        if base in Base.TWO_H_BONDS:
            element_3 = 2
        elif base in Base.THREE_H_BONDS:
            element_3 = 3
        elif base == Base.ZERO_PADDING:
            element_3 = 0
        else:
            raise ValueError(f"Unknown base: {base}")

        return [element_1, element_2, element_3]


WATSON_CRICK_PAIRS = [Base.A | Base.T,
                      Base.A | Base.U,
                      Base.C | Base.G]

WOBBLE_PAIRS = [Base.G | Base.U]


@dataclass
class BasePair:
    base_1: Base
    base_2: Base
    h_bonded: bool

    def __repr__(self):

        if self.base_1 == Base.ZERO_PADDING:
            repr_base_1 = ' '
        else:
            repr_base_1 = self.base_1.name

        if self.base_2 == Base.ZERO_PADDING:
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
        if self.base_1 | self.base_2 in WATSON_CRICK_PAIRS:
            return True

        return False

    def is_wobble(self):
        if self.base_1 | self.base_2 in WOBBLE_PAIRS:
            return True

        return False

    def to_vector(self, one_hot: bool = False,
                  pairing_type: PairingType = PairingType.STRUCTURE_BASED) -> list[int]:

        vector = base_to_vector(self.base_1, one_hot)
        vector.extend(base_to_vector(self.base_2, one_hot))
        if pairing_type == PairingType.WOBBLE_OR_WATSON_CRICK:

            if self.is_watson_crick() or self.is_wobble():
                vector.append(1)
            else:
                vector.append(0)

        elif pairing_type == PairingType.STRUCTURE_BASED:
            if self.h_bonded:
                vector.append(1)
            else:
                vector.append(0)

        elif pairing_type == PairingType.WATSON_CRICK:
            if self.is_watson_crick():
                vector.append(1)
            else:
                vector.append(0)
        else:
            raise ValueError(f"Pairing type must be structure-based, Watson-Crick, or a combination of Watson-Crick and Wobble. Got {pairing_type.name}")

        return vector
