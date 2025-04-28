from typing import Union

from mewtwo.embeddings.terminator.hairpin import RNAFoldHairpin, TransTermHPHairpin
from mewtwo.embeddings.terminator.a_tract import ATract
from mewtwo.embeddings.terminator.u_tract import UTract


class Terminator:
    def __init__(self, hairpin: Union[RNAFoldHairpin, TransTermHPHairpin], a_tract: ATract, u_tract: UTract):
        self.hairpin = hairpin
        self.a_tract = a_tract
        self.u_tract = u_tract

    def to_vector(self, one_hot: bool = False) -> list[int]:
        vector = []
        vector.extend(self.a_tract.sequence.to_vector(one_hot))
