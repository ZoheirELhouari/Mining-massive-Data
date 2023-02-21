import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
import random
from tqdm import tqdm

def load_data(data_zip_path):
    """Loading the data from the zip file."""
    # Unzip the dataset
    data_path = data_zip_path.replace(".zip", "")
    if not os.path.exists(data_path):
        print("Unzipping the dataset …", end=" ")
        with ZipFile(data_zip_path, 'r') as data_zip:
            data_zip.extractall("")
        print("Finished!")
    # Unless the dataset is unzipped
    else:
        print("The Dataset is already unzipped.")

    # Search for csv file 'small_dataset.csv', if it doesn't exist, it gets created 
    if not os.path.exists("small_dataset.csv"):
        print("Loading the small dataset …", end=" ")
         # Loading the features
        feats = pd.read_csv(data_path +
                            "/features.csv", index_col=0, header=[0, 1, 2])
        feats = feats.loc[tracks.index]
        # Loading the tracks
        tracks = pd.read_csv(data_path +
                             "/tracks.csv", index_col=0, header=[0, 1])
        tracks = tracks[tracks["set"]["subset"] == "small"]
        # Combining the tracks and features
        small_dataset = pd.concat([tracks["track"]["genre_top"],
                         tracks["set"]["split"], feats], axis=1)
        # Write on the csv file containing our small dataset 
        write_path = 'small_dataset.csv'
        small_dataset.to_csv(write_path)
        print(f"Finished! Written to file {write_path}!")
    # If not, load from directory 
    else:
        print("Loading dataset…")
        # reading the dataset csv file 
        small_dataset = pd.read_csv("small_dataset.csv", index_col=0, header=[0])
        print("Finished!")

    return small_dataset

def construct_random_Pmatrix(low_dim, feat_dim, norm=False):
    "Construction of the random projection matrix using Achlioptas method"
    
    projection_matrix = np.empty((low_dim, feat_dim))
    probabilities = [1/6, 2/3, 1/6]
    Values = [np.sqrt(3), 0, -np.sqrt(3)]

    for column in range(feat_dim):
        column_vec = random.choices(Values, probabilities, k=low_dim)
        if norm:
            vec_len = np.linalg.norm(column_vec)
            if vec_len:
                column_vec = column_vec / vec_len
        projection_matrix[:,column] = column_vec
        
    return projection_matrix

def euclidean_metric(vec1, vec2):
    """Calculating the euclidean distance between two feature vectors"""
    return np.linalg.norm(vec1-vec2)

def cosine_metric(vec1, vec2):
    """Calculating the cosine distance between two feature vectors"""
    return (np.dot(vec1.T, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))[0,0]

def generate_feature_matrix(data, spliting=["training"]):
    for sets in spliting:
        if sets not in ["training","test","validation"]:
          raise Exception("the data is splited into training, test and validation subsets, check your typing")
    splited_data = data[data.split.isin(spliting)]    
    return splited_data.drop(columns=["genre_top", "split"]).to_numpy().T  

def generate_feature_vector(trackID,data):  
    """returns the feature vector from  the original small dataset based on the track-ID."""
    return np.matrix(data.loc[trackID]).T
  


class HashTable:
    """Generates hashes in a table"""
    def __init__(self, hash_size, inp_dim):
        self.hash_table = {}
        "Construction of the random projection matrix using Achlioptas method"
        self.projections = construct_random_Pmatrix(hash_size, inp_dim) 
    
    def generate_hash(self, inp_vec):
        """Performs random projection method and converts projection to binary format"""
        bools = (np.dot(self.projections, inp_vec) > 0).astype(int)
        return "".join(bools.astype(str).tolist()) 
    
    def __setitem__(self, inp_vec, label):
        hash_val = self.generate_hash(inp_vec)
        self.hash_table[hash_val] = self.hash_table.get(hash_val, []) + [label]
        
    def __getitem__(self, inp_vec):
        hash_val = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_val, [])

class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.hash_tables = []
        self.results = None
        for i in range(num_tables):
            self.hash_tables.append(HashTable(hash_size, inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))
    
    def calculate_hashes(self,data, spliting=["training"]):
        # calaculates the hashes for each feature vector that coresponds to the selected spliting
        feature_matrix = generate_feature_matrix(data,spliting)
        # using tqdm library for the showing the progress bar 
        print("calculating the hashes")
        for i in tqdm(range(feature_matrix.shape[1])):
            self.__setitem__(feature_matrix[:,i], data.index[i])     

    
    def finding_similarities(self, data, spliting="validation"):
        # finding the similar tracks for each trackID in the training subset 
        validation_subset = generate_feature_matrix(data, spliting=[spliting])
        # creating a dataframe for storing our resutls 
        self.results = pd.DataFrame(index=data[data.split==spliting].index, 
                                    columns=["similar_tracks"], dtype=object)
        # using tqdm library for the showing the progress bar 
        print("finding the similarities")
        for i in tqdm(range(validation_subset.shape[1])):
            similar_tracks = self.__getitem__(validation_subset[:,i])
            self.results["similar_tracks"].iloc[i] = similar_tracks   
    
      

    def calculate_similarities(self, data):
        # droping genre_top and split columns so that we are only left with numerical values and no text 
        data = data.drop(columns=["genre_top", "split"])
        # creating empty columns to hold the values for cosine and euclidian similarities 
        self.results.insert(1,"cosine_similarities", [[] for l in range(self.results.shape[0])])
        self.results.insert(2,"euclidian_similarites", [[] for l in range(self.results.shape[0])])
        print("calculating the similarities")
        # using tqdm library for the showing the progress bar 
        for validation_trackID in tqdm(self.results.index):
            validation_vector = generate_feature_vector(validation_trackID, data)
            for similar_tracks in self.results["similar_tracks"].loc[validation_trackID]:
                similar_vec = generate_feature_vector(similar_tracks, data) 
                # Adding the similarities the to the dataframe
                self.results.at[validation_trackID, "cosine_similarities"].append((similar_tracks, cosine_metric(validation_vector, similar_vec)))
                self.results.at[validation_trackID, "euclidian_similarites"].append((similar_tracks, euclidean_metric(validation_vector, similar_vec)))
        print()    

    def k_nearest_neighbor(self,data,k):

        self.results.insert(1,"cosine_prediction", [[] for l in range(self.results.shape[0])])
        self.results.insert(4,"euclidian_prediction", [[] for l in range(self.results.shape[0])])
        # sorting the similarities 
        for index in tqdm(self.results.index):
            self.results.at[index, "cosine_similarities"].sort(key=lambda x: x[1], reverse=True)
            self.results.at[index, "euclidian_similarites"].sort(key=lambda x: x[1])   

        # we couldn't finish implementing the k_nearest neighbor du to the lack of contribution of a team member 



    def classification_accuracy(self,data):
        count_cosine = 0
        count_eucl = 0
        for track in data.index:
            if data.at[track, "genre_top"] == self.results.at[track, "cosine_prediction"]:
                count_cosine += 1
            if data.at[track, "genre_top"] == self.results.at[track, "euclidian_prediction"]:
                count_eucl += 1

        cos_accuracy = (count_cosine / data.shape[0]) * 100
        eucl_accuracy = (count_eucl / data.shape[0]) * 100

        print(cos_accuracy)
        print(eucl_accuracy)

        
    
    

small_dataset = load_data("fma_metadata.zip")
l = 120
n = 15


model = LSH(n, l, 518)
model.calculate_hashes(small_dataset, spliting=["training","validation"])

model.finding_similarities(small_dataset, spliting="test")
model.calculate_similarities(small_dataset)
k = 3
model.results
# model.k_nearest_neighbor(small_dataset,k)
# model.classification_accuracy(small_dataset)

     




