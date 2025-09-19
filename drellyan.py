import uproot
import numpy as np


class DrellYanDataset:
    def __init__(self, list_data_features, cut=None, min_mass=110, max_mass=160, sigma=5):
        file_path = (
            "/data/atlas/users/kdevries/hmumuml/RunIII/"
            "mc23_13p6TeV.700889.Sh_2214_Zmumu_mZ_105_ECMS_BFilter_HmumuSR_skimmed_prepared_FSR.root"
        )
        
        with uproot.open(file_path) as file_drell_yan:
            tree_drell_yan = file_drell_yan["tree_Hmumu"]

            # Higgs mass window cut
            cut_expression = f"(Truth_Boson_Mass >= {min_mass}) & (Truth_Boson_Mass <= {max_mass})"

            # Load requested features directly with cut applied
            features_to_load = list(set(list_data_features + ["Truth_Boson_Mass"]))
            data = tree_drell_yan.arrays(features_to_load, library="np", cut=cut_expression)
   
        # Stack features into matrix
        self.x = np.column_stack([data[feat] for feat in list_data_features]).astype(np.float32)

        # Filter outliers in one vectorized step
        self.filtered_x = self.remove_outliers_vectorized(self.x, sigma=sigma)
        self.N = self.filtered_x.shape[0]

        # Train/val/test split
        indices = np.arange(self.N)
        train_end = int(0.8 * self.N)
        val_end = int(0.9 * self.N)

        self.x_train = self.filtered_x[:train_end]
        self.x_val   = self.filtered_x[train_end:val_end]
        self.x_test  = self.filtered_x[val_end:]

        self.n_dims = self.x_train.shape[1]

    def remove_outliers_vectorized(self, data, sigma=5):
        """
        Remove rows where any feature is more than `sigma` std deviations from the mean.
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = (data - mean) / std
        mask = np.all(np.abs(z_scores) <= sigma, axis=1)
        return data[mask]

#
#class DrellYanDataset:
#    def __init__(self, list_data_features, cut=None):
#        #get file and tree 
#        file_drell_yan = uproot.open("/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700889.Sh_2214_Zmumu_mZ_105_ECMS_BFilter_HmumuSR_skimmed_prepared_FSR.root")
#        tree_drell_yan = file_drell_yan["tree_Hmumu"]
#
#        #Cutting Boson mass at 110-160 GeV to follow a close range around the higgs mass
#        cut_expression = f"{Truth_Boson_Mass} >= {110} &; {Truth_Boson_Mass} <= {160}"
#        filtered_data = tree_drell_yan.arrays(["Truth_Boson_Mass"], library="np", cut=cut_expression)
#        
#        # Get the indices of the filtered events
#        filtered_indices = np.where((filtered_data["Truth_Boson_Mass"] >= 110) &; (filtered_data["Truth_Boson_Mass"] <= 160))[0]
#        
#        # Extract the specified features from the filtered data and shape into float32
#        mu_drell_yan = tree_drell_yan.arrays(list_data_features, library="np")
#        self.x = np.column_stack([mu_drell_yan[key][filtered_indices] for key in list_data_features]).astype(np.float32)
#    
#        #filter data outliers
#        self.filtered_x = self.x.copy()
#        for i in range(len(list_data_features)):
#            self.filtered_x = self.remove_outliers(self.filtered_x, i)
#        self.N = self.filtered_x.shape[0]
#
#        # compute split sizes and split data
#        indices = np.arange(self.N)
#        train_end = int(0.8 * self.N)
#        val_end =   int(0.9 * self.N)
#
#        train_idx = indices[:train_end]
#        val_idx = indices[train_end:val_end]
#        test_idx = indices[val_end:]
#
#        self.x_train = self.filtered_x[train_idx]
#        self.x_val = self.filtered_x[val_idx]
#        self.x_test = self.filtered_x[test_idx]
#
#        self.n_dims = self.x_train.shape[1]
#
#
#    def remove_outliers(self, data, feature_idx, sigma=5):
#        """
#        Remove rows (events) where a given feature is more than 5 std deviations from the mean.
#
#        """
#        feature_values = data[:, feature_idx]
#        mean = np.mean(feature_values)
#        std = np.std(feature_values)
#
#        # boolean mask of events to keep
#        mask = np.abs(feature_values - mean) <= sigma * std
#
#        return data[mask]