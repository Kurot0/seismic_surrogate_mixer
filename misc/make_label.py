import os
import pickle
from itertools import cycle, islice
from collections import defaultdict
from sklearn.model_selection import train_test_split



def create_multishot_labels(output_file_path, directories):
    class_data = defaultdict(list)
    for folder_path in directories:
        for file in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file)):
                base_name = os.path.splitext(file)[0]
                class_id = base_name.split('-')[0].split('_')[1]
                class_data[class_id].append(base_name)

    class_groups = defaultdict(list)
    current_group_start = 0
    for class_id, data in class_data.items():
        num_groups = 10
        group_cycle = cycle(range(num_groups))
        group_cycle = islice(group_cycle, current_group_start, current_group_start + len(data))
        for idx, item in enumerate(data):
            group_number = next(group_cycle) % num_groups
            class_groups[group_number].append(item)
        current_group_start = (current_group_start + len(data)) % num_groups

    labels_dict = {}
    for fold in range(10):
        fold_key = f'quakeData-all-crossValidation{fold}'
        labels_dict[fold_key] = {'train': [], 'valid': [], 'test': []}

        valid_group = fold
        test_group = (fold + 1) % 10

        for group_number, group_data in class_groups.items():
            if group_number == valid_group:
                labels_dict[fold_key]['valid'].extend(group_data)
            elif group_number == test_group:
                labels_dict[fold_key]['test'].extend(group_data)
            else:
                labels_dict[fold_key]['train'].extend(group_data)

    with open(output_file_path, 'wb') as file:
        pickle.dump(labels_dict, file)


def create_zeroshot_labels(output_file_path, directories):
    class_data = defaultdict(list)
    for folder_path in directories:
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                base_name = os.path.splitext(file)[0]
                class_id = base_name.split('-')[0].split('_')[1]
                class_data[class_id].append(base_name)
    
    all_train_ids = [f"h{i:02d}" for i in range(1, 11)]
    
    zero_shot_ids = [f"h{i:02d}" for i in range(11, 17)]
    
    labels_dict = {}
    for fold_index, test_class_id in enumerate(zero_shot_ids):
        fold_key = f"quakeData-all-crossValidation{fold_index}"
        labels_dict[fold_key] = {'train': [], 'valid': [], 'test': []}
        
        test_data = class_data[test_class_id] if test_class_id in class_data else []
        
        trainvalid_ids = all_train_ids + [c for c in zero_shot_ids if c != test_class_id]
        
        trainvalid_data = []
        trainvalid_labels = []
        for c_id in trainvalid_ids:
            if c_id in class_data:
                for base_name in class_data[c_id]:
                    trainvalid_data.append(base_name)
                    trainvalid_labels.append(c_id)
        
        if len(trainvalid_data) > 0:
            train_data, valid_data = train_test_split(
                trainvalid_data,
                test_size=0.15,
                stratify=trainvalid_labels,
                random_state=42
            )
        else:
            train_data, valid_data = [], []
        
        labels_dict[fold_key]['train'].extend(train_data)
        labels_dict[fold_key]['valid'].extend(valid_data)
        labels_dict[fold_key]['test'].extend(test_data)
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(labels_dict, f)


# def create_zeroshot_labels2(output_file_path, directories):
#     class_data = defaultdict(list)
#     for folder_path in directories:
#         for file in os.listdir(folder_path):
#             full_path = os.path.join(folder_path, file)
#             if os.path.isfile(full_path):
#                 base_name = os.path.splitext(file)[0]
#                 class_id = base_name.split('-')[0].split('_')[1]
#                 class_data[class_id].append(base_name)
    
#     trainvalid_class_ids = [f"h{i:02d}" for i in range(1, 11)]
#     test_class_ids = [f"h{i:02d}" for i in range(11, 17)]
    
#     trainvalid_data = []
#     trainvalid_labels = []
#     for c_id in trainvalid_class_ids:
#         if c_id in class_data:
#             for base_name in class_data[c_id]:
#                 trainvalid_data.append(base_name)
#                 trainvalid_labels.append(c_id)
    
#     test_data = []
#     for c_id in test_class_ids:
#         if c_id in class_data:
#             test_data.extend(class_data[c_id])
    
#     train_data, valid_data = train_test_split(
#         trainvalid_data,
#         test_size=0.15,
#         stratify=trainvalid_labels,
#         random_state=42
#     )
    
#     fold_key = "quakeData-all-crossValidation0"  
#     labels_dict = {
#         fold_key: {
#             'train': train_data,
#             'valid': valid_data,
#             'test':  test_data
#         }
#     }
    
#     with open(output_file_path, 'wb') as f:
#         pickle.dump(labels_dict, f)


def main():
    directories = [
        'data/raw_data/2013_calc_scenario',
        'data/raw_data/source_R03',
        'data/raw_data/source_R04_done'
    ]

    create_multishot_labels('data/labels_dictionary.pkl', directories)
    create_zeroshot_labels('data/labels_dictionary_zeroshot.pkl', directories)
    # create_zeroshot_labels2('data/labels_dictionary_zeroshot2.pkl', directories)

if __name__ == "__main__":
    main()
