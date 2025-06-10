import os
import pickle
import torch


def generate_cv_data(x_data_path, y_data_path, labels_dict_path, cv_data_dir):
    if not os.path.exists(cv_data_dir):
        os.makedirs(cv_data_dir)

    with open(x_data_path, 'rb') as f:
        x_data = pickle.load(f)
    with open(y_data_path, 'rb') as f:
        y_data = pickle.load(f)
    with open(labels_dict_path, 'rb') as f:
        labels_dict = pickle.load(f)

    x_images = x_data['images']
    x_labels = x_data['labels']
    y_images = y_data['images']
    y_labels = y_data['labels']

    y_labels = [label.replace("Sv3", "sd1") for label in y_labels]
    y_labels = [label.replace("_mod", "") for label in y_labels]

    fold_keys = sorted(labels_dict.keys())

    for i, fold_key in enumerate(fold_keys):
        dir_name = os.path.join(cv_data_dir, f'crossValidation_all{i}')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for phase in ['train', 'valid', 'test']:
            phase_labels = labels_dict[fold_key][phase]

            x_indexes = [index for index, label in enumerate(x_labels) if label in phase_labels]
            y_indexes = [index for index, label in enumerate(y_labels) if label in phase_labels]

            x_phase_images = torch.tensor(x_images[x_indexes])
            y_phase_images = torch.tensor(y_images[y_indexes])

            torch.save(x_phase_images, os.path.join(dir_name, f'x_{phase}.pt'), pickle_protocol=4)
            torch.save(y_phase_images, os.path.join(dir_name, f'y_{phase}.pt'), pickle_protocol=4)

            print(f'[LabelDict={os.path.basename(labels_dict_path)}, Fold={i}, Phase={phase}] '
                  f'x_shape: {x_phase_images.shape}, y_shape: {y_phase_images.shape}')

def main():
    x_data_path = 'data/prep_data/source.pkl'
    y_data_path = 'data/prep_data/Sv_LL.pkl'

    pairs = [
        ('data/labels_dictionary.pkl', 'data/exp_data/cv_data'),
        ('data/labels_dictionary_zeroshot.pkl', 'data/exp_data/cv_data_zeroshot'),
        # ('data/labels_dictionary_zeroshot2.pkl', 'data/exp_data/cv_data_zeroshot2')
    ]

    for labels_dict_path, cv_data_dir in pairs:
        print(f'=== Processing: {labels_dict_path} --> {cv_data_dir} ===')
        generate_cv_data(x_data_path, y_data_path, labels_dict_path, cv_data_dir)

if __name__ == "__main__":
    main()
