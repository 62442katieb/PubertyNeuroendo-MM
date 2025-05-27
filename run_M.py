'''
Run separate models for male and female participants, so I’ve attached two pickle files with the synthetic data, 
along with lists of the input features and the hormones we’re trying to predict from them. 
Note that the output hormone lists are different for male and female models.

Author: Jinyao Tian
Date: 2025-04-28
'''

import model
import pickle


def main():
    # Load the male pickle file
    with open('synthetic_dset-M.pkl', 'rb') as f:
        m_data = pickle.load(f)

    # Read variable names from txt file
    with open('variable_names.txt', 'r') as file:
        content = file.read()

    # Split content into list definitions
    list_definitions = content.split('\n\n')  # Assuming lists are separated by blank lines

    # Initialize empty lists
    input_features = []
    hormones_F = []
    hormones_M = []

    # Parse each list
    for definition in list_definitions:
        if 'input_features = [' in definition:
            # Extract items between brackets
            items = definition.split('[')[1].split(']')[0].strip()
            input_features = [item.strip(' ",') for item in items.split('\n') if item.strip(' ",')]
        elif 'hormones-F = [' in definition:
            items = definition.split('[')[1].split(']')[0].strip()
            hormones_F = [item.strip(' ",') for item in items.split('\n') if item.strip(' ",')]
        elif 'hormones-M = [' in definition:
            items = definition.split('[')[1].split(']')[0].strip()
            hormones_M = [item.strip(' ",') for item in items.split('\n') if item.strip(' ",')]

    # Now you can use these lists
    print(f"Number of input features: {len(input_features)}")
    print(f"Number of female hormones: {len(hormones_F)}")
    print(f"Number of male hormones: {len(hormones_M)}")

    '''
    Focus on the male model first.
    The model will be trained to predict the following hormones:
        1. "filtered_dhea",
        2. "filtered_ert",
    '''

    # Run
    model.train(dset = m_data, 
                features = input_features, 
                phenotypes = hormones_M,
            runs=1, epochs=2, seed=4,
            n_nodes=[64, 32, 16],
            out_dir='./train_output',
            across_dataset=False,  # pass these in if you need them
            exp_dataset=False)

if __name__ == "__main__":
    # Needed on macOS / Windows when num_workers > 0
    main()