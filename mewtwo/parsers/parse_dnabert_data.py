def parse_dnabert_data(input_file: str) -> tuple[list[str], list[float]]:
    seqs = []
    tes = []
    with open(input_file, 'r') as seq_data:
        for line in seq_data:
            line = line.strip()
            seq, te = line.split('\t')
            te = float(te)
            seqs.append(seq)
            tes.append(te / 100)
    return seqs, tes
