import os
import torch


SEQ_LEN = 12                
FORECAST_LEN = 12 #4            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MACRO_COLUMNS = ['GDP', 'INDPRO', 'PSAVERT', 'CPIAUCSL', 'DFF',
       'MRTSSM44000USS', 'UNEMPLOY', 'GDP_DIFF_Q', 'GDP_DIFF_Y',
       'INDPRO_DIFF_Q', 'INDPRO_DIFF_Y', 'PSAVERT_DIFF_Q', 'PSAVERT_DIFF_Y',
       'CPIAUCSL_DIFF_Q', 'CPIAUCSL_DIFF_Y', 'DFF_DIFF_Q', 'DFF_DIFF_Y',
       'MRTSSM44000USS_DIFF_Q', 'MRTSSM44000USS_DIFF_Y', 'UNEMPLOY_DIFF_Q',
       'UNEMPLOY_DIFF_Y']

SIZE_METRIC_COLUMN = 'Total Revenue' 
LATENT_FIN_DIM_HYP = 75 
LATENT_MACRO_DIM_HYP = 75

TICKERS = ['COHR',
 'ENFY',
 'ETR',
 'FNMA',
 'FNMAG',
 'FNMAH',
 'FNMAI',
 'FNMAJ',
 'FNMAK',
 'FNMAL',
 'FNMAM',
 'FNMAN',
 'FNMAO',
 'FNMAP',
 'FNMAS',
 'FNMAT',
 'FNMFN',
 'FNMFO',
 'FSFG',
 'MKSI',
 'ORCL',
 'OSBC',
 'SFBS',
 'T',
 'T-PA',
 'T-PC',
 'TBB',
 'XOM']