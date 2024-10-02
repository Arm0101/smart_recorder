import os
from utils import convert_audios_to_wav


def rename_files_in_directory(directory_path, new_name_prefix):
    if not os.path.exists(directory_path):
        return

    files = os.listdir(directory_path)
    for i, filename in enumerate(files):
        old_file_path = os.path.join(directory_path, filename)

        if os.path.isfile(old_file_path):
            new_file_name = f"{new_name_prefix}_{i}{os.path.splitext(filename)[1]}"

            new_file_path = os.path.join(directory_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renombrado: {filename} -> {new_file_name}")


rename_files_in_directory("../test/unknown/", "unknown")
rename_files_in_directory("../new_dataset/carlos", "carlos")
convert_audios_to_wav('../new_dataset/speaker_1')
convert_audios_to_wav('../new_dataset/speaker_2')
convert_audios_to_wav('../new_dataset/speaker_3')
convert_audios_to_wav('../new_dataset/speaker_4')
