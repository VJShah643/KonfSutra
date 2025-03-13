import os

#read file from path and output the content
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
