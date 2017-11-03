class ParamConfig:
    def __init__(self):
        self.root_dir = "../.."
        self.data_folder = "%s/Data" % self.root_dir
        self.feat_folder = "%s/Feat" % self.root_dir
        self.original_train_data_path = "%s/train.csv" % self.data_folder
        self.original_test_data_path = "%s/test.csv" % self.data_folder
        self.processed_train_data_path = "%s/train_processd.csv" % self.data_folder
        self.processed_test_data_path = "%s/test_processd.csv" % self.data_folder
        #self.train_features_path = "%s/train_features.csv" % self.feat_folder
        #self.test_features_path = "%s/test_features.csv" % self.feat_folder
        self.random_seed = 1024 + 1010

config = ParamConfig()

