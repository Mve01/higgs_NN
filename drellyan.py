import uproot
import numpy as np


batch_size = 1000

class DrellYanDataset:
    def __init__(self):
        #get file and tree 
        file_drell_yan = uproot.open("/data/atlas/users/kdevries/hmumuml/RunIII/mc23_13p6TeV.700889.Sh_2214_Zmumu_mZ_105_ECMS_BFilter_HmumuSR_skimmed_prepared_FSR.root")
        tree_drell_yan = file_drell_yan["tree_Hmumu"]

        #get specific data
        list_data_features = ["Muons_Eta_Lead", "Muons_Eta_Sub", "Muons_PT_Lead", "Muons_PT_Sub", "Muons_Phi_Lead", "Muons_Phi_Sub"]
        mu_drell_yan = tree_drell_yan.arrays(list_data_features, library = "np")

        #shape data into float32 for easier and optimal use
        self.x = np.column_stack([mu_drell_yan[key] for key in mu_drell_yan.keys()]).astype(np.float32)

        #filter data outliers
        self.filtered_x = self.x.copy()
        for i in range(len(list_data_features)):
            self.filtered_x = self.remove_outliers(self.filtered_x, i)
        self.N = self.filtered_x.shape[0]

        # compute split sizes and split
        indices = np.arange(self.N)
        train_end = int(0.8 * self.N)
        val_end = int(0.9 * self.N)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # split the data 
        self.x_train = self.filtered_x[train_idx]
        self.x_val = self.filtered_x[val_idx]
        self.x_test = self.filtered_x[test_idx]

        self.n_dims = self.x_train.shape[1]


    def remove_outliers(self, data, feature_idx, sigma=5):
        """
        Remove rows (events) where a given feature is more than 5 std deviations from the mean.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features)
        feature_idx : int
            Index of the feature to check (e.g., PT_Lead)
        sigma : float
            Number of standard deviations for outlier cutoff

        Returns
        -------
        filtered_data : np.ndarray
            Data with outlier events removed
        """
        feature_values = data[:, feature_idx]
        mean = np.mean(feature_values)
        std = np.std(feature_values)

        # boolean mask of events to keep
        mask = np.abs(feature_values - mean) <= sigma * std

        # apply mask to all features (remove whole event)
        filtered_data = data[mask]
        return filtered_data