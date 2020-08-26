import os
path = 'result/gray'
for name in os.listdir(path):
    folder_path = os.path.join(path,name)
    if os.path.isdir(folder_path):
        if len([lists for lists in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, lists))])!=7:
            weight_dir = os.path.join('weights','/'.join(folder_path.split('/')[1:]))
            print(weight_dir)

# print('filenum:',len([lists for lists in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, lists))]))