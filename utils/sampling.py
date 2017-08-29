import numpy as np
import sys
import os
from PIL import Image
from random import randint
import math
import cv2
from .utilities import *
from .params import *

class GetSample():
	def __init__(self):
		self.pos_face_dir = 'Test_source/face/'
		self.pos_ppl_dir = 'Test_source/ppl/'
		self.neg_dir = 'Test_source/bg/'

	def get_pos_img(self, dim):
		self.sanity_check(self.pos_face_dir, self.pos_ppl_dir)
		print("Samples Checked..")
		list_pos_face = [elem for elem in os.listdir(self.pos_face_dir)]
		list_pos_ppl = [elem for elem in os.listdir(self.pos_ppl_dir)]
		pos_db_12 = [0 for _ in range(len(list_pos_face))]

		for i in range(len(list_pos_face)):
			sys.stdout.write("\rIMG_INDEX:" + str(i+1) + '/' + str(len(list_pos_face)))
			# Find the ground true bounding box and store it
			face_arr, _, _ = face_crop(self.pos_face_dir + list_pos_face[i])
			ymin = face_arr[0]
			xmin = face_arr[1]
			ymax = face_arr[2]
			xmax = face_arr[3]

			img = Image.open(self.pos_ppl_dir + list_pos_ppl[i])
			pos_db_line_12 = np.zeros((6, dim, dim, 3))
			cropped_img = img.crop((xmin, ymin, xmax, ymax))
			cropped_img_90 = cropped_img.transpose(Image.ROTATE_90)
			cropped_img_270 = cropped_img.transpose(Image.ROTATE_270)
			
			flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
			flipped_img_90 = flipped_img.transpose(Image.ROTATE_90)
			flipped_img_270 = flipped_img.transpose(Image.ROTATE_270)

			cropped_arr_12 = img2array(cropped_img, dim)
			cropped_arr_90 = img2array(cropped_img_90, dim)
			cropped_arr_270 = img2array(cropped_img_270, dim)
			
			flipped_arr_12 = img2array(flipped_img, dim)
			flipped_arr_90 = img2array(flipped_img_90, dim)
			flipped_arr_270 = img2array(flipped_img_270, dim)

			pos_db_line_12[0,:] = cropped_arr_12
			pos_db_line_12[1,:] = cropped_arr_90
			pos_db_line_12[2,:] = cropped_arr_270
			pos_db_line_12[3,:] = flipped_arr_12
			pos_db_line_12[4,:] = flipped_arr_90
			pos_db_line_12[5,:] = flipped_arr_270

			pos_db_12[i] = pos_db_line_12

			img.close(), cropped_img.close(), flipped_img.close()

		pos_db_12 = [elem for elem in pos_db_12 if type(elem) != int]
		pos_db_12 = np.vstack(pos_db_12)
		assert(pos_db_12.shape[1]==dim) and pos_db_12.shape[2]==dim and pos_db_12.shape[3]==3
		print('\nFinished')

		return pos_db_12

	def get_neg_img(self, dim, Trans=False):
		"""
			Trans: create two set of array which is generate by dim/2 and dim respectively. The previous one is
			feed to the (lower) 12 Det-CNN (if dim = 24) to evaluate how lower CNN perform in selected samples
		"""
		neg_file_list = [elem for elem in os.listdir(self.neg_dir)]
		neg_db_12 = [0 for _ in range(len(neg_file_list))]
		
		# Another set of list to store the smaller version
		if Trans:
			new_dim = int(dim/2)
			neg_db_12_sub = [0 for _ in range(len(neg_file_list))]

		for i in range(len(neg_file_list)):
			img = Image.open(self.neg_dir + neg_file_list[i])

			# Set the maximum size bound for neg_img
			if img.size[0] > p_neg_max_bound or img.size[1] > p_neg_max_bound:
				ratio = p_neg_max_bound / max(img.size[0], img.size[1])
				img = img.resize((int(ratio * img.size[0]), int(ratio * img.size[1])))
			neg_db_line = np.zeros((p_neg_per_img, dim, dim, 3), np.float32)

			if Trans:
				neg_db_line_sub = np.zeros((p_neg_per_img, new_dim, new_dim, 3), np.float32)

			for neg_iter in range(p_neg_per_img):
				rad_rand = randint(dim, min(img.size[0]-1, img.size[1]-1, p_neg_threshold))
				while(rad_rand<=p_face_minimum):
					rad_rand = randint(0, min(img.size[0]-1, img.size[1]-1, p_neg_threshold))
				x_rand = randint(0, img.size[0] - rad_rand)
				y_rand = randint(0, img.size[1] - rad_rand)

				cropped_img = img.crop((x_rand, y_rand, x_rand + rad_rand, y_rand + rad_rand))
				cropped_arr_12 = img2array(cropped_img, dim)

				if Trans:
					cropped_arr_12_sub = img2array(cropped_img, new_dim)
					assert(cropped_arr_12_sub.shape[0] == dim/2) and cropped_arr_12_sub.shape[1] == dim/2 and cropped_arr_12_sub.shape[2] == 3

				neg_db_line[neg_iter,:] = cropped_arr_12

				if Trans:
					neg_db_line_sub[neg_iter,:] = cropped_arr_12_sub

				cropped_img.close()

			neg_db_12[i] = neg_db_line

			if Trans:
				neg_db_12_sub[i] = neg_db_line_sub
			img.close()

		neg_db_12 = [elem for elem in neg_db_12 if type(elem) != int]
		neg_db_12 = np.vstack(neg_db_12)
		neg_db_12 = neg_db_12[:int(neg_db_12.shape[0] / p_neg_batch) * p_neg_batch,:]

		if Trans:
			neg_db_12_sub = [elem for elem in neg_db_12_sub if type(elem) != int]
			neg_db_12_sub = np.vstack(neg_db_12_sub)
			neg_db_12_sub = neg_db_12_sub[:int(neg_db_12_sub.shape[0] / p_neg_batch) * p_neg_batch,:]

			return neg_db_12, neg_db_12_sub
		else:
			return neg_db_12

	def sample_neg_amount(self, dim, amount, Trans=False):
		if Trans:
			samp, _ = self.get_neg_img(dim, True)
			length = samp.shape[0]
			assert(length < amount)
			iters = int(np.ceil(amount / length))
			neg_db = [0 for _ in range(iters)]
			neg_db_sub = [0 for _ in range(iters)]
			for i in range(iters):
				sys.stdout.write("\rProgress: " + str(i+1) + "/" + str(int(amount/length) + 1))
				subsets, subsets_sub = self.get_neg_img(dim, True)
				neg_db[i] = subsets
				neg_db_sub[i] = subsets_sub
			neg_db = np.vstack(neg_db)
			neg_db_sub = np.vstack(neg_db_sub)
			return neg_db[:amount], neg_db_sub[:amount]		

		else:
			length = self.get_neg_img(dim).shape[0]
			assert (length < amount)
			iters = int(amount / length) + 1
			neg_db = [0 for _ in range(iters)]
			for i in range(iters):
				subsets = self.get_neg_img(dim)
				neg_db[i] = subsets
				sys.stdout.write("\rProgress: " + str(i+1) + "/" + str(int(amount/length) + 1))
			neg_db = np.vstack(neg_db)
			return neg_db[:amount]



	def sanity_check(self, path1, path2):
	    list_pos_face = [os.path.splitext(item)[0] for item in os.listdir(self.pos_face_dir)]
	    list_pos_ppl = [os.path.splitext(item)[0] for item in os.listdir(self.pos_ppl_dir)]
	    assert(len(list_pos_face) == len(list_pos_ppl))
	    error = 0
	    for i in range(len(list_pos_face)):
	        if list_pos_face[i] != list_pos_ppl[i]:
	            error = 1
	            break
	    assert(error==0)
	    pass		

class SampleEval():
	def pred_in_step(net, input_nodes, inputs, step):
		"""
			Get the confidence level of given CNN for each crop image.
		"""
		assert(inputs.shape[0] >= step)
		iters = int(np.ceil(inputs.shape[0]/step))
		store_list = [0 for _ in range(iters)]
		start = 0
		for it in range(iters):
			start = step * it
			end = start + step
			#print(start,end, it)
			pred_matrix = net.prediction_flatten.eval(feed_dict={input_nodes:inputs[start:end,:]})
			store_list[it] = pred_matrix
			sys.stdout.write('\rProgress:' + str(it+1) + '/' + str(iters))
		total_arr = np.vstack(store_list)
		assert(total_arr.shape[0] == inputs.shape[0])
		return total_arr

	def neg_img_nextnet(dim, amount, net, input_nodes, step, thres=1e-2):
		"""
			Args: 
				param1 (int): dim of intended CNN
				param2 (tf.net obj): tensorflow trained neural net
				param3 (tf.placeholder): tf.placeholder for inputs
				param4 (np.array): sampled dimxdimx3 colored image crop in np.arr format
				param5 (int): the amount of evaluation each around
				param6 (float): threshold for filter

			Returns:
				np.array: float number indicate the confidence level of CNN for each crop image

			Filter out the negative crop sample which has confidence level lower than 0.01.
			Preserve only crops which has confidence level greater than 0.01

		"""
		get = GetSample()
		total_list = []
		cnt = len(total_list)
		while(cnt < amount):
			print('Current Amount: ',cnt)
			neg_db_24, neg_db_24_sub = get.sample_neg_amount(dim, 100000, Trans=True)
			pred_matrix = SampleEval.pred_in_step(net, input_nodes, neg_db_24_sub, step)
			filter_idx = np.where(pred_matrix>thres)[0]
			neg_db_24_filter = neg_db_24[filter_idx]
			total_list.append(neg_db_24_filter)
			cnt += neg_db_24_filter.shape[0]
		return total_list