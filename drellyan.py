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
        train_end = int(0.7 * self.N)
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
