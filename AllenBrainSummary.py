import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nilearn import datasets, plotting

RECEPTORS = [
    #'ESR1',
    'ESR2',
    'GPER',
    'AR'
]

allen_brain_all = '/Users/katherine.b/Dropbox/Projects/kangaroo/meta-matching/AllenBrain-ER1_ER2_AR_GPER-ReceptorsAll.csv'

er_all = pd.read_csv(allen_brain_all, index_col=[0,1], header=0)
er_all['Brain Region'] = er_all['structure_abbreviation'].replace(
    {
        'AMY': 'Amyg',
        'MFC': 'rACC/mPFC',
        'CBC': 'Crbllm',
        'DFC': 'dlPFC',
        'HIP': 'Hipp',
        'ITC': 'ITG, BA20',
        'MD': 'MDThal',
        'STC': 'pSTG',
        'IPC': "IPL",
        'A1C': "Aud",
        'S1C': "S1",
        'M1C': "M1",
        'V1C': 'V1',
        'VFC': 'vlPFC',
        'STR': "Striatum"
    }
)

for i in er_all.index:
    num, unit = er_all.loc[i]['donor_age'].split(' ')
    num = int(num)
    if unit == 'pcw':
        er_all.at[i, 'Donor Age'] = 0 - (num / 52)
        er_all.at[i, 'Life Stage'] = 'prenatal'
    elif unit == 'mos':
        er_all.at[i, 'Donor Age'] = num / 12
        er_all.at[i, 'Life Stage'] = 'infant'
    elif unit == 'yrs':
        er_all.at[i, 'Donor Age'] = num
        if num < 9:
            er_all.at[i, 'Life Stage'] = 'child'
        elif num < 11:
            er_all.at[i, 'Life Stage'] = 'pre puberty'
        elif num < 14:
            er_all.at[i, 'Life Stage'] = 'puberty'
        elif num < 20:
            er_all.at[i, 'Life Stage'] = 'post puberty'
        elif num < 30:
            er_all.at[i, 'Life Stage'] = 'young adulth'
        elif num < 55:
            er_all.at[i, 'Life Stage'] = 'adult'
        else:
            er_all.at[i, 'Life Stage'] = 'older adult'

teen_regions = list(er_all[er_all['Donor Age'].between(8,14, inclusive='both')]['structure_abbreviation'].unique())
puberty_brains = er_all[er_all['Donor Age'].between(8,15, inclusive='both')]
adult_brains = er_all[er_all['Donor Age'].between(24,60, inclusive='both')]

region_order = [
    'vlPFC',
    'rACC/mPFC',
    'dlPFC',
    'OFC',
    'Amyg',
    'Hipp',
    "Striatum",
    'MDThal',
    'ITG, BA20',
    'pSTG',
    "IPL",
    "Aud",
    "S1",
    "M1",
    'V1',
    'Crbllm',
]

fig,ax = plt.subplots(
    nrows=2, 
    ncols=3, 
    figsize=(10.5,5), 
    sharex=True,
    #layout='tight'
)
for receptor in RECEPTORS:
    temp = RECEPTORS.index(receptor)
    plt.tight_layout(h_pad=1)
    temp_ax = ax[0,temp]
    sns.boxplot(
        puberty_brains,
        x='Brain Region',
        y=receptor,
        order=region_order,
        ax=temp_ax
    )
    temp_ax.axhline(
        puberty_brains[receptor].median(), 
        linestyle='--', 
        color='#888888'
    )
    temp_ax.set_xlabel('')
    if temp == 0:
        temp_ax.set_title('Early Adolescence', loc='left')
    #### adults #############
    temp_ax = ax[1,temp]
    if temp == 0:
        temp_ax.set_title('Adulthood', loc='left')
    sns.boxplot(
        adult_brains,
        x='Brain Region',
        y=receptor,
        order=region_order,
        ax=temp_ax
    )
    temp_ax.axhline(
        adult_brains[receptor].median(), 
        linestyle='--', 
        color='#888888'
    )
    for label in temp_ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')
    sns.despine()
fig.savefig(
    'receptors_by_region.png',
    facecolor='#FFFFFF',
    dpi=400,
    bbox_inches='tight'
)