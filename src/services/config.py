"""Constants shared by the modelling code."""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: tickers left out of every sample (set during the first analyses of the data; kept so that the sample does not change)
EXCLUDED_TICKERS = ["COHR", "ENFY", "ETR", "FNMA", "FNMAG", "FNMAH", "FNMAI", "FNMAJ", "FNMAK", "FNMAL", "FNMAM", "FNMAN",
                    "FNMAO", "FNMAP", "FNMAS", "FNMAT", "FNMFN", "FNMFO", "FSFG", "MKSI", "ORCL", "OSBC", "SFBS", "T",
                    "T-PA", "T-PC", "TBB", "XOM"]
