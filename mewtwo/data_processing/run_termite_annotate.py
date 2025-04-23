import os
from pprint import pprint

from mewtwo.data_processing.iterate_over_dir import iterate_over_dir

#
# def get_file_sets(bigwig_folder, termite_folder):
#     sample_id_to_files = {}
#     for folder_name, folder_path in iterate_over_dir(bigwig_folder, get_dirs=True):
#         for sample_name, _ in iterate_over_dir(folder_path, '.bw'):
#             strand = sample_name.split('_')[-1]
#             sample_id = '_'.join(sample_name.split('_')[:-1])
#             if sample_id not in sample_id_to_files:
#                 sample_id_to_files[sample_id] = {'forward_bw': '',
#                                                  'reverse_bw': '',
#                                                  'forward_termite': '',
#                                                  'reverse_termite': ''}
#             sample_id_to_files[sample_id][f'{strand}_bw'] = f"{sample_id}:/data/bigwig/{folder_name}/{sample_name}.bw"
#
#     for narrowpeak_name, _ in iterate_over_dir(termite_folder, '.narrowPeak'):
#         if narrowpeak_name.startswith('3prime'):
#             sample_id = '_'.join(narrowpeak_name.split('_')[:3])
#             strand = narrowpeak_name.split('_')[3]
#             sample_id_to_files[sample_id][f'{strand}_termite'] = f"/data/termite/{narrowpeak_name}.narrowPeak"
#
#     return sample_id_to_files


def get_file_sets(bigwig_folder, termite_folder):
    sample_id_to_files = {}
    for sample_name, folder_path in iterate_over_dir(bigwig_folder, get_dirs=True):
        sample_id = '_'.join(sample_name.split('_')[:2])
        if sample_id not in sample_id_to_files:
            sample_id_to_files[sample_id] = {'forward_bw': '',
                                             'reverse_bw': '',
                                             'forward_termite': '',
                                             'reverse_termite': ''}
        for replicate_name, _ in iterate_over_dir(folder_path, '.bw'):
            strand = sample_name.split('_')[-1]

            if not sample_id_to_files[sample_id][f'{strand}_bw']:
                sample_id_to_files[sample_id][f'{strand}_bw'] = f"{sample_id}:/data/bigwig/{sample_name}/{replicate_name}.bw"
            else:
                sample_id_to_files[sample_id][f'{strand}_bw'] += f",/data/bigwig/{sample_name}/{replicate_name}.bw"

    for narrowpeak_name, _ in iterate_over_dir(termite_folder, '.narrowPeak'):
        if narrowpeak_name.startswith('3prime'):
            strand = narrowpeak_name.split('_')[-1]
            sample_id = '_'.join(narrowpeak_name.split('_')[3:5])
            sample_id_to_files[sample_id][f'{strand}_termite'] = f"/data/termite/{narrowpeak_name}.narrowPeak"

    return sample_id_to_files


def run_termite_annotate(input_folder: str):
    bigwig_folder = os.path.join(input_folder, 'bigwig')
    genome_folder = os.path.join(input_folder, 'genome')
    termite_folder = os.path.join(input_folder, 'termite')

    genome = None
    gff = None

    for file_name, file_path in iterate_over_dir(genome_folder, '.fasta'):
        genome = f"/data/genome/{file_name}.fasta"

    assert genome

    for file_name, file_path in iterate_over_dir(genome_folder, '.gff3'):
        gff = f"/data/genome/{file_name}.gff3"

    assert gff

    sample_id_to_files = get_file_sets(bigwig_folder, termite_folder)
    sample_ids = list(sample_id_to_files.keys())
    sample_ids.sort()

    print(sample_ids)

    termite_forward_string = []
    termite_reverse_string = []
    bw_forward_string = []
    bw_reverse_string = []
    sample_string = []

    for sample_id in sample_ids:
        termite_forward_string.append(sample_id_to_files[sample_id]["forward_termite"])
        termite_reverse_string.append(sample_id_to_files[sample_id]["reverse_termite"])
        bw_forward_string.append(sample_id_to_files[sample_id]["forward_bw"])
        bw_reverse_string.append(sample_id_to_files[sample_id]["reverse_bw"])
        sample_string.append(sample_id)

    termite_forward_string = ' '.join(termite_forward_string)
    termite_reverse_string = ' '.join(termite_reverse_string)
    bw_forward_string = ' '.join(bw_forward_string)
    bw_reverse_string = ' '.join(bw_reverse_string)
    sample_string = ' '.join(sample_string)



    docker_command = f"docker run --rm -v $(pwd):/data --platform=linux/amd64 termite annotate --termite-out-forward {termite_forward_string} --termite-out-reverse {termite_reverse_string} --sample-names {sample_string} --rna-3prime-ends-forward {bw_forward_string} --rna-3prime-ends-reverse {bw_reverse_string} --genome {genome} --gene-annotations {gff} --output /data/termite_annotate/all_runs --trans-term-hp --rnafold --upstream-nt 100 --downstream-nt 20"
    os.system(docker_command)


if __name__ == "__main__":
    run_termite_annotate(os.getcwd())