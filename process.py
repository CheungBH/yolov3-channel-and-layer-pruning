# allocate train and validation images
import glob, os

current_dir = "data/jump/JPEGImages"
print(current_dir)

percentage_test = 10

file_train = open('data/jump/jump_train.txt', 'w')
file_test = open('data/jump/jump_val.txt', 'w')

counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        file_test.write(current_dir + "/" + title + ".jpg" + "\n")
    else:
        file_train.write(current_dir + "/" + title + ".jpg" + "\n")
        counter = counter + 1
