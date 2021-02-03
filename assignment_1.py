# =====================================================================
# assignment_1.py
#
# Adapted from https://github.com/akcarsten/fMRI_data_analysis
# Visit the repository for further detail on and analysis of fMRI data
#
# Preliminaries:
# - install python 3.{5,6,7}
# - install pip: https://www.makeuseof.com/tag/install-pip-for-python/
# - install pip packages:
#   - pip install -U requests numpy nibabel matplotlib tqdm
#
# Helpful resources:
# - matplotlib user's guide: https://matplotlib.org/users/index.html
# - NumPy user guide: https://docs.scipy.org/doc/numpy/user/
# =====================================================================
# For py2/3 compat/cleanliness:
from __future__ import print_function, absolute_import, division

import os
import sys
import zipfile
import requests
import numpy as np
import nibabel
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the data url and paths
URL = ('http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/'
       'MoAEpilot.zip')
DATA_DIR = './fMRI_data'
ZIP_FILENAME = 'data.zip'
S_DATA_DIR = 'sM00223'  # Structural data directory
F_DATA_DIR = 'fM00223'  # Functional data directory


# =======================================
# START ASSIGNMENT 1 HERE
# -----------------------
# A "TODO" indicates a place for you to
# add or modify code.
# =======================================
def main():
    # =======================================
    # Let's start with some basics
    # Recommended reading:
    #     https://www.scipy-lectures.org/intro/numpy/operations.html
    # =======================================

    # Vectorization example, ApA = A + A
    A = [[5, 9], [-5, 7]]
    ApA = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            ApA[i][j] = A[i][j] + A[i][j]
    print('ApA', ApA)

    # This can be rewritten more efficiently (using NumPy) as:
    A = np.array([[5, 9], [-5, 7]])
    ApA = A + A
    print('ApA', ApA)

    # Now look at this next unvectorized multiplication:
    A = [[5, 9], [-5, 7]]
    A_squared = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            A_squared[i][j] = A[i][j] * A[i][j]

    print('A_squared', A_squared)

    A = np.array([[5, 9], [-5, 7]])
    # *=====================================*
    # TODO: Compute A_squared vectorized
    # (change None to the correct expression)
    # *=====================================*
    A_squared = A*A

    print('A_squared', A_squared)

    # ---------------
    # Creating and adding vectors
    A = np.array([1, 1, 2, 3, 5, 8])
    B = np.ones_like(A)
    # *=====================================*
    # TODO: Compute C as the sum of A and B
    # (change None to the correct expression)
    # *=====================================*
    C = A + B

    if C is None:
        sys.exit(1)
    
    # Printing shapes and NumPy arrays
    print('A.shape', A.shape)
    print('A', A)
    print('B.shape', B.shape)
    print('B', B)
    print('C.shape', C.shape)
    print('C', C)
    
    # Check your answer
    assert np.all(C == np.array([2, 2, 3, 4, 6, 9])), 'Incorrect sum for `C`!'
    
    # ---------------
    # Creating and adding matrices
    A = np.array([[2, 9, 1],
                  [0, 4,-2],
                  [1,-1, 8]])
    # *=====================================*
    # TODO: Create a B matrix from a random
    # normal distribution of the same shape
    # as A (change None to the correct
    # expression)
    # *=====================================*
    B = np.shape(np.random.normal())
    
    if B is None:
        sys.exit(1)

    print('B.shape', B.shape)
      
    # *=====================================*
    # TODO: Compute C as sum of A and B
    # *=====================================*
    C = A + B

    # ---------------
    # Computing a dot product
    A = np.array([[2, 9, 1],
                  [0, 4,-2],
                  [1,-1, 8]])
    B = np.array([[5,-1, 0],
                  [1,-2, 0],
                  [9, 6, 1]])
    # *=====================================*
    # TODO: Compute C as dot product of A and
    # B. Check out the documentation on the
    # NumPy dot function (np.dot)
    # *=====================================*
    C = np.dot(A,B)

    if C is None:
        sys.exit(1)
    
    assert np.all(C == np.array([[ 28, -14,   1],
                                 [-14, -20,  -2],
                                 [ 76,  49,   8]])), 'Incorrect dot product!'

    # ---------------
    # Python notebooks on solving an ODE with Euler's approximation (read first-order only)
    # http://sam-dolan.staff.shef.ac.uk/mas212/notebooks/ODE_Example.html
    # https://ipython-books.github.io/123-simulating-an-ordinary-differential-equation-with-scipy/
    
    # =======================================
    # Begin the fMRI visualization portion of
    # the assignment.
    # Download and extract the fMRI data
    # =======================================

    # Download the "Auditory - single subject" dataset
    dl_path = os.path.join(DATA_DIR, ZIP_FILENAME)
    if not os.path.isfile(dl_path):
        # Function defined below main()
        download_file(URL, dl_path)

    # Un-zip the data
    with zipfile.ZipFile(dl_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # =======================================
    # Open and visualize the structural data
    # =======================================

    # Find all files in the structural data folder
    data_path = os.path.join(DATA_DIR, S_DATA_DIR)
    files = os.listdir(data_path)

    # Read in the structural data (first one in HDR format)
    data = None
    for data_file in files:
        if data_file.lower().endswith('.hdr'):
            data = nibabel.load(os.path.join(data_path, data_file)).get_data()
            break

    # Exit if no data found...
    if data is None:
        print('No ".hdr" data found in "{}", exiting.'.format(data_path))
        sys.exit(1)

    # Remove 4th dimension (empty) and rotate 90 degrees
    data = np.rot90(data.squeeze(), 1)

    # *=======================================*
    # TODO: Modify the following code to show
    # every 5th slice of the scan. Plot the
    # slices in 12 plots over 2 rows, 6 per
    # row.
    # *=======================================*
    # <[M1] Modification starts here>

    # Plot every 10th slice of the scan
    fig, ax = plt.subplots(1, 6, figsize=[18, 3])

    n = 0
    slice_ = 0  # slice is a Python built-in, append '_' to avoid overriding
    for _ in range(6):
        # Confused by the colons? Check out the NumPy docs on indexing:
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        ax[n].imshow(data[:, :, slice_], 'gray')
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        ax[n].set_title('Slice number: {}'.format(slice_), color='C0')
        n += 1
        slice_ += 12

    # Why tight_layout?
    # A reason why: https://stackoverflow.com/questions/9603230/
    fig.tight_layout()
    plt.show()

    # <[M1] Modification ends here>

    # =======================================
    # Open and visualize the functional data
    # =======================================

    # Find all files in the data folder
    data_path = os.path.join(DATA_DIR, F_DATA_DIR)
    files = os.listdir(data_path)

    # Read in the data and organize it with respect to the acquisition
    # parameters
    data_all = []
    for data_file in files:
        if data_file.lower().endswith('.hdr'):
            data = nibabel.load(os.path.join(data_path, data_file)).get_data()
            data_all.append(data.squeeze())

    # Exit if no data found...
    if not data_all:
        print('No ".hdr" data found in "{}", exiting.'.format(data_path))
        sys.exit(1)

    # Convert Python list to NumPy array
    data_all = np.asarray(data_all)

    # Create a 3x6 subplot
    fig, ax = plt.subplots(3, 6, figsize=[18, 11])

    # Organize the data for visualisation in the coronal plane
    coronal = np.transpose(data_all, [1, 3, 2, 0])
    coronal = np.rot90(coronal, 1)

    # Organize the data for visualisation in the transversal plane
    transversal = np.transpose(data_all, [2, 1, 3, 0])
    transversal = np.rot90(transversal, 2)

    # Organize the data for visualisation in the sagittal plane
    sagittal = np.transpose(data_all, [2, 3, 1, 0])
    sagittal = np.rot90(sagittal, 1)

    # Plot some of the images in different planes
    n = 10
    for i in range(6):
        ax[0][i].imshow(coronal[:, :, n, 0], cmap='gray')
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        if i == 0:
            ax[0][i].set_ylabel('coronal', fontsize=25, color='C0')
        n += 10

    n = 5
    for i in range(6):
        ax[1][i].imshow(transversal[:, :, n, 0], cmap='gray')
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        if i == 0:
            ax[1][i].set_ylabel('transversal', fontsize=25, color='C0')
        n += 10

    n = 5
    for i in range(6):
        ax[2][i].imshow(sagittal[:, :, n, 0], cmap='gray')
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        if i == 0:
            ax[2][i].set_ylabel('sagittal', fontsize=25, color='C0')
        n += 10

    fig.tight_layout()
    plt.show()

    # =======================================
    # Visualize the signal strength of a
    # voxel over time
    # =======================================

    # Create an empty plot with defined aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])

    # Plot a random voxel over time (96 time steps)
    voxel_data = transversal[30, 30, 35, :]

    ax.plot(voxel_data, lw=3)
    ax.set_xlim([0, transversal.shape[3] - 1])
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Signal Strength', fontsize=20)
    # *=======================================*
    # TODO: Add a fitting title to the plot
    # *=======================================*
    # <Your code here>
    plt.title('Time vs Signal Strength')
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    # *=======================================*
    # TODO: Show the third plot
    # *=======================================*
    # <Your code here>


    # *=======================================*
    # TODO: Write code to plot a histogram of
    # the voxel's signal strength (contained
    # in voxel_data). See the matplotlib docs
    # on the "hist" function. Remember to
    # LABEL your axes!!!
    # *=======================================*
    # <Your code here>
    plt.hist(vowel_data)
    plt.xlabel("Time")
    plt.ylabel("Signal Strength")
    plt.title("Signal Strenght Histogram")
    plt.show()


# Helper function
def download_file(url, path, progress=sys.stdout, chunk_size=4096):
    # Get the director(ies) of the provided path
    dirname = os.path.dirname(path)

    # Check if the target folder for storing the data already exists. If not
    # create it and save the zip file.
    if not os.path.isdir(dirname):
        if os.path.isfile(dirname):
            raise FileExistsError('Specified path contains an uncreated '
                                  'directory with the same name as an existing '
                                  'file: {}'.format(dirname))
        os.makedirs(dirname)

    # Write ('w') the binary data ('b') to path
    with open(path, 'wb') as f:
        print('Downloading "{}" to "{}" from "{}."'.format(
            os.path.basename(path), os.path.abspath(dirname), url))

        # Get the response (streaming to print progress)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            with tqdm(total=int(total_length), unit_divisor=1024, miniters=1,
                      file=progress or sys.stdout, unit_scale=True, unit='B',
                      disable=progress is False) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    bar.update(len(data))
                    f.write(data)

            # Flush buffer
            if hasattr(progress, 'flush'):
                progress.flush()


# It is good practice in Python to include this snippet in your scripts
# See: https://thepythonguru.com/what-is-if-__name__-__main__/
if __name__ == '__main__':
    print("Executing main program")
    main()
