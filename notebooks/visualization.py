# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# Code for visualization
# index = np.random.randint(0,len(data))
# fname = data.loc[index,'fname']
# im = np.array(Image.open(r'C:\Users\Edoardo\PycharmProjects\Car_Prediction\data\cars_train\{fname}'.format(fname=fname)), dtype=np.uint8)
# fig, ax = plt.subplots(1)
# ax.imshow(im)
# rect = patches.Rectangle((data.iloc[index,1],data.iloc[index,2]),data.iloc[index,3]-data.iloc[index,1],data.iloc[index,4]-data.iloc[index,2],linewidth=1,edgecolor='r',facecolor='none')
# ax.add_patch(rect)
# plt.show()