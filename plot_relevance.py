'''
Hi it's Katie :))))
'''

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns

from nilearn import datasets, plotting

from os.path import join

input_features = [
    "smri_area_cdk_banksstslh",
    "smri_area_cdk_cdacatelh",
    "smri_area_cdk_cdmdfrlh",
    "smri_area_cdk_cuneuslh",
    "smri_area_cdk_ehinallh",
    "smri_area_cdk_fusiformlh",
    "smri_area_cdk_ifpllh",
    "smri_area_cdk_iftmlh",
    "smri_area_cdk_ihcatelh",
    "smri_area_cdk_locclh",
    "smri_area_cdk_lobfrlh",
    "smri_area_cdk_linguallh",
    "smri_area_cdk_mobfrlh",
    "smri_area_cdk_mdtmlh",
    "smri_area_cdk_parahpallh",
    "smri_area_cdk_paracnlh",
    "smri_area_cdk_parsopclh",
    "smri_area_cdk_parsobislh",
    "smri_area_cdk_parstgrislh",
    "smri_area_cdk_pericclh",
    "smri_area_cdk_postcnlh",
    "smri_area_cdk_ptcatelh",
    "smri_area_cdk_precnlh",
    "smri_area_cdk_pclh",
    "smri_area_cdk_rracatelh",
    "smri_area_cdk_rrmdfrlh",
    "smri_area_cdk_sufrlh",
    "smri_area_cdk_supllh",
    "smri_area_cdk_sutmlh",
    "smri_area_cdk_smlh",
    "smri_area_cdk_frpolelh",
    "smri_area_cdk_tmpolelh",
    "smri_area_cdk_trvtmlh",
    "smri_area_cdk_insulalh",
    "smri_area_cdk_banksstsrh",
    "smri_area_cdk_cdacaterh",
    "smri_area_cdk_cdmdfrrh",
    "smri_area_cdk_cuneusrh",
    "smri_area_cdk_ehinalrh",
    "smri_area_cdk_fusiformrh",
    "smri_area_cdk_ifplrh",
    "smri_area_cdk_iftmrh",
    "smri_area_cdk_ihcaterh",
    "smri_area_cdk_loccrh",
    "smri_area_cdk_lobfrrh",
    "smri_area_cdk_lingualrh",
    "smri_area_cdk_mobfrrh",
    "smri_area_cdk_mdtmrh",
    "smri_area_cdk_parahpalrh",
    "smri_area_cdk_paracnrh",
    "smri_area_cdk_parsopcrh",
    "smri_area_cdk_parsobisrh",
    "smri_area_cdk_parstgrisrh",
    "smri_area_cdk_periccrh",
    "smri_area_cdk_postcnrh",
    "smri_area_cdk_ptcaterh",
    "smri_area_cdk_precnrh",
    "smri_area_cdk_pcrh",
    "smri_area_cdk_rracaterh",
    "smri_area_cdk_rrmdfrrh",
    "smri_area_cdk_sufrrh",
    "smri_area_cdk_suplrh",
    "smri_area_cdk_sutmrh",
    "smri_area_cdk_smrh",
    "smri_area_cdk_frpolerh",
    "smri_area_cdk_tmpolerh",
    "smri_area_cdk_trvtmrh",
    "smri_area_cdk_insularh",
    "smri_thick_cdk_banksstslh",
    "smri_thick_cdk_cdacatelh",
    "smri_thick_cdk_cdmdfrlh",
    "smri_thick_cdk_cuneuslh",
    "smri_thick_cdk_ehinallh",
    "smri_thick_cdk_fusiformlh",
    "smri_thick_cdk_ifpllh",
    "smri_thick_cdk_iftmlh",
    "smri_thick_cdk_ihcatelh",
    "smri_thick_cdk_locclh",
    "smri_thick_cdk_lobfrlh",
    "smri_thick_cdk_linguallh",
    "smri_thick_cdk_mobfrlh",
    "smri_thick_cdk_mdtmlh",
    "smri_thick_cdk_parahpallh",
    "smri_thick_cdk_paracnlh",
    "smri_thick_cdk_parsopclh",
    "smri_thick_cdk_parsobislh",
    "smri_thick_cdk_parstgrislh",
    "smri_thick_cdk_pericclh",
    "smri_thick_cdk_postcnlh",
    "smri_thick_cdk_ptcatelh",
    "smri_thick_cdk_precnlh",
    "smri_thick_cdk_pclh",
    "smri_thick_cdk_rracatelh",
    "smri_thick_cdk_rrmdfrlh",
    "smri_thick_cdk_sufrlh",
    "smri_thick_cdk_supllh",
    "smri_thick_cdk_sutmlh",
    "smri_thick_cdk_smlh",
    "smri_thick_cdk_frpolelh",
    "smri_thick_cdk_tmpolelh",
    "smri_thick_cdk_trvtmlh",
    "smri_thick_cdk_insulalh",
    "smri_thick_cdk_banksstsrh",
    "smri_thick_cdk_cdacaterh",
    "smri_thick_cdk_cdmdfrrh",
    "smri_thick_cdk_cuneusrh",
    "smri_thick_cdk_ehinalrh",
    "smri_thick_cdk_fusiformrh",
    "smri_thick_cdk_ifplrh",
    "smri_thick_cdk_iftmrh",
    "smri_thick_cdk_ihcaterh",
    "smri_thick_cdk_loccrh",
    "smri_thick_cdk_lobfrrh",
    "smri_thick_cdk_lingualrh",
    "smri_thick_cdk_mobfrrh",
    "smri_thick_cdk_mdtmrh",
    "smri_thick_cdk_parahpalrh",
    "smri_thick_cdk_paracnrh",
    "smri_thick_cdk_parsopcrh",
    "smri_thick_cdk_parsobisrh",
    "smri_thick_cdk_parstgrisrh",
    "smri_thick_cdk_periccrh",
    "smri_thick_cdk_postcnrh",
    "smri_thick_cdk_ptcaterh",
    "smri_thick_cdk_precnrh",
    "smri_thick_cdk_pcrh",
    "smri_thick_cdk_rracaterh",
    "smri_thick_cdk_rrmdfrrh",
    "smri_thick_cdk_sufrrh",
    "smri_thick_cdk_suplrh",
    "smri_thick_cdk_sutmrh",
    "smri_thick_cdk_smrh",
    "smri_thick_cdk_frpolerh",
    "smri_thick_cdk_tmpolerh",
    "smri_thick_cdk_trvtmrh",
    "smri_thick_cdk_insularh",
    "smri_vol_scs_crbcortexlh",
    "smri_vol_scs_tplh",
    "smri_vol_scs_caudatelh",
    "smri_vol_scs_putamenlh",
    "smri_vol_scs_pallidumlh",
    "smri_vol_scs_bstem",
    "smri_vol_scs_hpuslh",
    "smri_vol_scs_amygdalalh",
    "smri_vol_scs_aal",
    "smri_vol_scs_vedclh",
    "smri_vol_scs_crbcortexrh",
    "smri_vol_scs_tprh",
    "smri_vol_scs_caudaterh",
    "smri_vol_scs_putamenrh",
    "smri_vol_scs_pallidumrh",
    "smri_vol_scs_hpusrh",
    "smri_vol_scs_amygdalarh",
    "smri_vol_scs_aar",
    "smri_vol_scs_vedcrh",
]

hormones_F = [
    "filtered_dhea",
    "filtered_ert",
    "filtered_hse",
]

hormones_M = [
    "filtered_dhea",
    "filtered_ert",
]

dhea_rel = "/Users/katherine.b/Dropbox/Projects/kangaroo/meta-matching/train_output/feature_importance_ig/run0_target0_relevance.npy"
ert_rele = "/Users/katherine.b/Dropbox/Projects/kangaroo/meta-matching/train_output/feature_importance_ig/run0_target1_relevance.npy"



PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"
RESOURCE = "/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/resources/"

DK_TO_ABCD_L = pd.read_csv(
    join(RESOURCE, 'dk_to_abcd.csv'), 
    header=0, index_col=0
)
DK_TO_ABCD_R = pd.read_csv(
    join(RESOURCE, 'dk_to_abcd_r.csv'), 
    header=0, index_col=0
)

surface_r = nib.load(join(RESOURCE, 'fs_LR.32k.R.pial.surf.gii'))
surface_l = nib.load(join(RESOURCE, 'fs_LR.32k.L.pial.surf.gii'))

# DK atlas
desikan_r = nib.load(join(RESOURCE, 'Desikan.32k.R.label.gii'))
desikan_l = nib.load(join(RESOURCE, 'Desikan.32k.L.label.gii'))
# subcortical atlas
subcortic = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
subcort_array = subcortic.maps.get_fdata()


dhea_vals = pd.Series(
    np.load(dhea_rel),
    index=input_features
)

ert_vals = pd.Series(
    np.load(ert_rele),
    index=input_features
)

temp_thk = desikan_r.agg_data()
dhea = np.zeros_like(temp_thk, dtype=np.float32)
right_temp = dhea_vals.filter(like='rh')
for i in range(1,36):
    if i != 4:
        # get region abbrev
        rgn = DK_TO_ABCD_R.loc[i]['thk_var'].split('_')[-1][:-2]
        # get region value
        val = dhea_vals.loc[f'smri_thick_cdk_{rgn}rh']
        dhea = np.where(temp_thk == i, val.astype(np.float32), dhea)
