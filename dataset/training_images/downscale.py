import os
import glob
from PIL import Image
from PIL.Image import NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX, HAMMING

hr_img_list = glob.glob(os.path.join("training_hr_images", "*"))

#print(hr_img_list)



# trim the hr img size to multiple of 3
for i in hr_img_list:
	basename = os.path.basename(i)
	print("----------------------file name: ", basename, " ---------------------------")
	hr_img = Image.open(i)
	# tuple are immutable
	old_size = list(hr_img.size)
	print("old size: ", old_size)

	# Check if any dim num < 48x3 (144), 這樣downsacle完才會還有48
	flag = 0
	for i in range(2):
		if old_size[i] < 144:
			flag = 1
			old_size[i] = 144
	if flag == 1:
		hr_img = hr_img.resize(old_size, resample=BICUBIC)


	# trim
	for i in range(2):
		trim = old_size[i] % 3
		print(trim)
		old_size[i] = old_size[i] - trim 
	hr_trim_img = hr_img.resize(old_size, resample=BICUBIC)
	new_size = hr_trim_img.size
	print("new size:", new_size)	
	hr_trim_img.save(os.path.join("training_hr_trim_images", basename))




	# Resize (3 倍downscale)
	lr_img = hr_trim_img.resize( (new_size[0]//3, new_size[1]//3) , resample=BICUBIC)
	print("lr size: ", lr_img.size)
	lr_img.save(os.path.join("training_lr_images", basename))





	