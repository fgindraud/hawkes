import os

def DirectoryOfThisScript():
    return os.path.dirname(os.path.abspath(__file__))

project_root = DirectoryOfThisScript()

def FlagsForFile( filename, **kwargs ):
    # Flags from the Makefile
    return {
            'flags': ['-x', 'c++', '-std=c++17', '-Wall', '-Wextra'],
            'include_paths_relative_to_dir': project_root,
            'override_filename': project_root + "/src/main.cpp"
            }
