import os


def iterate_over_dir(directory, extension=None, get_dirs=False):
    for file_name in os.listdir(directory):
        if not get_dirs:
            if file_name.endswith(extension):
                file_label = file_name.split(extension)[0]
                file_path = os.path.join(directory, file_name)
                yield file_label, file_path
        else:
            file_path = os.path.join(directory, file_name)
            if os.path.isdir(file_path):
                yield file_name, file_path
