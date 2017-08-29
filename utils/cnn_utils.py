import tensorflow as tf
from PIL import Image
from skimage.transform import pyramid_gaussian
import numpy as np
from utils.params import *
import math

def img_scalar(path):
	img = Image.open(path)
	if img.size[0] > p_neg_max_bound or img.size[1] > p_neg_max_bound:
		ratio = p_neg_max_bound / max(img.size[0], img.size[1])
		resized = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
		img.close()
		return resized
	else:
		return img

def img_scalar_raw(img):
    if img.size[0] > p_neg_max_bound or img.size[1] > p_neg_max_bound:
        ratio = p_neg_max_bound / max(img.size[0], img.size[1])
        resized = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
        img.close()
        return resized
    else:
        return img

def index_top(arr, top_num=5):
	arr = arr.reshape(-1)
	new_arr = np.zeros(45)
	new_arr[:] = arr
	index_list = []
	for i in range(top_num):
		if i >= 1 and max(new_arr) / (arr[index_list[0]]) < 0.01:
			break
		idx = np.argmax(new_arr)
		index_list.append(idx)
		#print(max(new_arr))
		new_arr[idx] = 0.0
	return index_list

def detection_windows(img, net_type, detect_net, input_node, thres=1e-2):
	# generate pyramid of images from sample image
	pyramid = tuple(pyramid_gaussian(img, downscale=1.18))
	pyramid_list = [0 for _ in range(len(pyramid))]
	for scale in range(len(pyramid_list)):
		# Read the scaled image and resize accordingly
		_x = pyramid[scale]
		resized = Image.fromarray(np.uint8(_x*255)).resize((int(_x.shape[1]*float(net_type/p_face_minimum)), 
                                                            int(_x.shape[0]*float(net_type/p_face_minimum))))
    	# Break if scaled img is less than selected net input size
		if resized.size[0] < net_type or resized.size[1] < net_type:
			break
        # Convert back to 0-1 scale
		resized = np.asarray(resized).astype(np.float32)/255    
		resized_row = resized.shape[0]
		resized_col = resized.shape[1]
		# sanity check
		if resized.shape[2] > 3:
			resized = resized[:,:,:3]
	    # reshape array
		resized = np.reshape(resized,(1,resized_row,resized_col,3))
	    # Calculate the possible number of sliding windows
		win_num_row = math.floor((resized_row-net_type)/4)+1
		win_num_col = math.floor((resized_col-net_type)/4)+1
	    # Generate the list of result from CNN
		result = detect_net.prediction.eval(feed_dict={input_node:resized})
		# Re-calibrate the size of result matrices
		result = result[:,\
	                :win_num_row,\
	                :win_num_col,\
	                :]
		assert(result.shape[2] == win_num_col)
		result = result.reshape(-1,1) 
		# Return the box where only the confidence score is greater than 0.01
		result_index = np.where(result>thres)[0] 
		# Store selected detection windows and resize them to fit the original img
		bounding_box = np.zeros((len(result_index), 5))
		bounding_box[:,0] = (result_index % win_num_col) * 4
   
		bounding_box[:,1] = (result_index / win_num_col) * 4
		bounding_box[:,2] = bounding_box[:,0] + net_type - 1
		bounding_box[:,3] = bounding_box[:,1] + net_type - 1
		bounding_box[:,4] = result[result_index,0]

		bounding_box[:,0] = bounding_box[:,0] / resized_col * img.size[0]
		bounding_box[:,1] = bounding_box[:,1] / resized_row * img.size[1]
		bounding_box[:,2] = bounding_box[:,2] / resized_col * img.size[0]
		bounding_box[:,3] = bounding_box[:,3] / resized_row * img.size[1]

		bounding_box = bounding_box.tolist()

		bounding_box = [elem + [img.crop([int(elem[0]),
								int(elem[1]),
								int(elem[2]),
								int(elem[3])]), scale] for id_,elem in enumerate(bounding_box)]
		if len(bounding_box) > 0:
			pyramid_list[scale] = bounding_box
	pyramid_list = [elem for elem in pyramid_list if type(elem) != int]
	result_box = [pyramid_list[i][j] for i in range(len(pyramid_list)) for j in range(len(pyramid_list[i]))]
	img.close()

	return result_box


def detection_windows_ver2(img, net_type, detect_net, input_node, keep_prob, thres=200):
	# generate pyramid of images from sample image
	pyramid = tuple(pyramid_gaussian(img, downscale=1.18))
	scale_list = []
	for id_, py in enumerate(pyramid):
		if min(py.shape[0], py.shape[1]) < (min(img.size[0], img.size[1])*12/36.0) and min(py.shape[0], py.shape[1]) > 12:
			scale_list.append(id_)

	pyramid_list = [0 for _ in range(len(scale_list))]
	for scale in scale_list:
		# Read the scaled image and resize accordingly
		_x = pyramid[scale]
		resized = Image.fromarray(np.uint8(_x*255)).resize((int(_x.shape[1]*float(net_type/p_face_minimum)), 
                                                            int(_x.shape[0]*float(net_type/p_face_minimum))))

        # Convert back to 0-1 scale
		resized = np.asarray(resized).astype(np.float32)/255    
		resized_row = resized.shape[0]
		resized_col = resized.shape[1]

		# sanity check
		if resized.shape[2] > 3:
			resized = resized[:,:,:3]
	    # reshape array
		resized = np.reshape(resized,(1,resized_row,resized_col,3))
	    # Calculate the possible number of sliding windows
		win_num_row = math.floor((resized_row-12)/4)+1
		win_num_col = math.floor((resized_col-12)/4)+1
		if win_num_row < 1 or win_num_col < 1:
			break
	    # Generate the list of result from CNN
		result = detect_net.prediction.eval(feed_dict={input_node:resized,keep_prob:1.0})
		# Re-calibrate the size of result matrices
		result = result[:,\
	                :win_num_row,\
	                :win_num_col,\
	                :]        
		assert(result.shape[2] == win_num_col)
		result = result.reshape(-1,1) 
		# Return the box where only the confidence score is greater than 0.01
		result_index = np.where(result>0)[0] 
		# Store selected detection windows and resize them to fit the original img
		bounding_box = np.zeros((len(result_index), 5))
		bounding_box[:,0] = (result_index % win_num_col) * 4
   
		bounding_box[:,1] = (result_index / win_num_col) * 4
		bounding_box[:,2] = bounding_box[:,0] + 12 - 1
		bounding_box[:,3] = bounding_box[:,1] + 12 - 1
		bounding_box[:,4] = result[result_index,0]

		bounding_box[:,0] = bounding_box[:,0] / resized_col * img.size[0]
		bounding_box[:,1] = bounding_box[:,1] / resized_row * img.size[1]
		bounding_box[:,2] = bounding_box[:,2] / resized_col * img.size[0]
		bounding_box[:,3] = bounding_box[:,3] / resized_row * img.size[1]

		bounding_box = bounding_box.tolist()

		bounding_box = [elem + [img.crop([int(elem[0]),
								int(elem[1]),
								int(elem[2]),
								int(elem[3])]), scale] for id_,elem in enumerate(bounding_box)]
		if len(bounding_box) > 0:
			pyramid_list[scale-scale_list[0]] = bounding_box
	pyramid_list = [elem for elem in pyramid_list if type(elem) != int]
	result_box = [pyramid_list[i][j] for i in range(len(pyramid_list)) for j in range(len(pyramid_list[i]))]
	result_box = sorted(result_box, key = lambda x:x[4], reverse=True)
	result_box = result_box[:thres]
	img.close()

	return result_box

def det_box_eval(face_arr, result_box):
	score_list = [0 for _ in range(len(result_box))]
	real_area = (face_arr[2] - face_arr[0]) * (face_arr[3] - face_arr[1])
	#print("real_area: ", real_area)
	for id_, window in enumerate(result_box):
		eval_minx = max(face_arr[0], window[0])
		eval_miny = max(face_arr[1], window[1])
		eval_maxx = min(face_arr[2], window[2])
		eval_maxy = min(face_arr[3], window[3])
		#print(eval_minx, eval_miny, eval_maxx, eval_maxy)

		win_area = (window[2] - window[0]) * (window[3] - window[1])
		#print("win: ", win_area)
		eval_area = (eval_maxx - eval_minx) * (eval_maxy - eval_miny)
		#print("eval_area: ", eval_area)

		
		if (eval_area / real_area) < .4 or (win_area / real_area) >= 1.8 or (eval_minx >= eval_maxx) or (eval_miny >= eval_maxy):
			score_list[id_] = -1
		else:
			score_area = eval_area / win_area
			score_list[id_] = score_area

	pos_index = np.where(np.asarray(score_list)>=0.3)[0]
	return pos_index

def det_window_calibration(detec_win, cali_dict, pred_calib, top_num=3):
	uncali_arr = np.asarray(detec_win[:4])
	scale = detec_win[6]
	idx_list = index_top(pred_calib, top_num)
	ss, xx, yy = 0, 0, 0
	for i in range(len(idx_list)):
		ss += cali_dict[idx_list[i]][0] / float(len(idx_list))
		xx += cali_dict[idx_list[i]][1] / float(len(idx_list))
		yy += cali_dict[idx_list[i]][2] / float(len(idx_list))
		#ss, xx, yy = cali_dict[_id][0], cali_dict[_id][1], cali_dict[_id][2]

	xmin_new = int(uncali_arr[0] - (uncali_arr[2]-uncali_arr[0])*xx / ss)
	ymin_new = int(uncali_arr[1] - (uncali_arr[3]-uncali_arr[1])*yy / ss)
	xmax_new = int(xmin_new + (uncali_arr[2]-uncali_arr[0]) / ss)
	ymax_new = int(ymin_new + (uncali_arr[3]-uncali_arr[1]) / ss)
	#print(ss, xx, yy)
	#print(pred_calib)
	return np.array([xmin_new, ymin_new, xmax_new, ymax_new, scale])

def det_window_calibration_24after(detec_win, cali_dict, pred_calib, top_num=3):
	uncali_arr = np.asarray(detec_win[:4])
	scale = detec_win[4]
	idx_list = index_top(pred_calib, top_num)
	ss, xx, yy = 0, 0, 0
	for i in range(len(idx_list)):
		ss += cali_dict[idx_list[i]][0] / float(len(idx_list))
		xx += cali_dict[idx_list[i]][1] / float(len(idx_list))
		yy += cali_dict[idx_list[i]][2] / float(len(idx_list))
		#ss, xx, yy = cali_dict[_id][0], cali_dict[_id][1], cali_dict[_id][2]

	xmin_new = int(uncali_arr[0] - (uncali_arr[2]-uncali_arr[0])*xx / ss)
	ymin_new = int(uncali_arr[1] - (uncali_arr[3]-uncali_arr[1])*yy / ss)
	xmax_new = int(xmin_new + (uncali_arr[2]-uncali_arr[0]) / ss)
	ymax_new = int(ymin_new + (uncali_arr[3]-uncali_arr[1]) / ss)
	#print(ss, xx, yy)
	#print(pred_calib)
	return np.array([xmin_new, ymin_new, xmax_new, ymax_new, scale])

def NMS(boxes, overlapGate):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    xmin = boxes[:,0]
    ymin = boxes[:,1]
    xmax = boxes[:,2]
    ymax = boxes[:,3]
    
    area = (xmax-xmin+1) * (ymax-ymin+1)
    idxs = np.argsort(ymax)
    
    chosen_idxs = []
    while len(idxs)>0: 
        end = len(idxs) - 1
        current = idxs[end]
        chosen_idxs.append(current)
        
        xminx = np.maximum(xmin[current], xmin[idxs[:end]])
        yminy = np.maximum(ymin[current], ymin[idxs[:end]])
        xmaxx = np.minimum(xmax[current], xmax[idxs[:end]])
        ymaxy = np.minimum(ymax[current], ymax[idxs[:end]])
        
        bound_width = np.maximum(0, xmaxx - xminx + 1)
        bound_height = np.maximum(0, ymaxy - yminy + 1)
        
        overlap = (bound_width * bound_height) / area[idxs[:end]]
        
        delete_idxs = np.concatenate(([end], np.where(overlap>overlapGate)[0]))
        idxs = np.delete(idxs, delete_idxs)
        
    return boxes[chosen_idxs].astype(np.int32)


def NMS_by_scale(boxes_with_scale, overlapGate):
	NMS_12_adjusted = [0 for _ in range(len(np.unique(boxes_with_scale[:,4])))]
	
	for id_,scale in enumerate(np.unique(boxes_with_scale[:,4])):    
		idxs = np.where(boxes_with_scale[:,4] == scale)
		NMS_subset = NMS(boxes_with_scale[idxs], 0.5)
		NMS_12_adjusted[id_] = NMS_subset
	
	NMS_12_adjusted = np.concatenate([arr for arr in NMS_12_adjusted])
	return NMS_12_adjusted