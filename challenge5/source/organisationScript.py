os.popen('echo "hello"')
labels = load_labels(labelsPath=labelsPath, classes = CLASSES)
# print(labels[0])
    
for i in range(len(labels)):
    oldImage = imagesTrainPath + "Image_" + str(i+1) + ".png"
    newImage = imagesTrainPath + str(labels[i]) +  "/" + "Image_" + str(i+1) + ".png"
    subprocess.call(["cp", oldImage, newImage])