import os
import tarfile
from multiprocessing import Pool

def extract_member(member, tar_dir, extract_dir):
    with tarfile.open(tar_dir, 'r:gz') as tar:
        tar.extract(member, path=extract_dir)

def extract_all(tar_dir, extract_dir):
    with tarfile.open(tar_dir, 'r:gz') as tar:
        members = tar.getmembers()
    nprocs = max(1, os.cpu_count()// 2) 
    with Pool(nprocs) as pool:
        pool.starmap(extract_member, [(member, tar_dir, extract_dir) for member in members])

# Example usage:
extract_all('fineweb.tar.gz', './edu_fineweb10')