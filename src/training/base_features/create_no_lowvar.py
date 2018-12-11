"""Creates numpy array of rescaled gene expression values, excluding low-variance genes with Var<.1"""

import os
from sklearn import feature_selection as fsel
from sklearn import preprocessing as prep
import numpy as np
import helpers.data_load as dl

print(os.getcwd())

# Loading file
print('Loading Trainset')
DATA_FOLDER = '../../../data/'
RES_FOLDER = '../../../results/base_features/'

train_x, gene_names_x, cell_names_x = dl.load_data(DATA_FOLDER + 'train_data.csv.gz')

# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
train_x = train_x[:, 1:]
gene_names_x = gene_names_x[1:]
cell_names_x = cell_names_x[1:]

train_y, cell_names_y = dl.load_response(DATA_FOLDER + 'response.csv.gz')

# Preprocessing
varThresh = fsel.VarianceThreshold(threshold=0.1).fit(train_x)
scaled_x = prep.scale(varThresh.transform(train_x))

print('Loading Herring 2017')
herring_x, gene_names_herring, cell_names_herring = dl.load_data(DATA_FOLDER + 'herring2017_data.csv.gz')
# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
herring_x = herring_x[:, 1:]
gene_names_herring = gene_names_herring[1:]
herring_scaled = prep.scale(varThresh.transform(herring_x))

# Load Joost 2016 Data
print('Loading Joost 2016')
joost_x, gene_names_joost, cell_names_joost = dl.load_data(DATA_FOLDER + 'joost2016_data.csv.gz')
# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
joost_x = joost_x[:, 1:]
#gene_names_joost= gene_names_joost[1:]
joost_scaled = prep.scale(varThresh.transform(joost_x))

# Save
np.save(RES_FOLDER+'train_x_rescaled_nolovar.npy', scaled_x)
np.save(RES_FOLDER+'herring2017_rescaled_nolovar.npy', herring_scaled)
np.save(RES_FOLDER+'joost2016_rescaled_nolovar.npy', joost_scaled)
