#!/usr/bin/python

#This is a simple implementation of the naive bayes classifier
#author: Arvind RS (arvindrs.gb@gmail.com)
#date: 2016/09/12

#MIT License

#Copyright (c) [year] [fullname]

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import csv, random, math, os
from operator import itemgetter
import numpy as np 
from decimal import Decimal,getcontext
import time

smoothing_param = 1.0

def load_data(filename):

	feat_counts = {}

	word_count = 0
	line_count = 0
	with open(filename, 'r') as file:
		for line in file:
			line_count += 1
			token = line.split(" ")
			for feature in token[0:-1]:
				word_count += 1
				feature_name, counts = feature.split(":")
				if feature_name not in feat_counts:
					feat_counts[feature_name] = 0
				feat_counts[feature_name] += int(counts)

	# remove all features that occur less than 3 (threshold) times
	to_remove = []
	for key, value in feat_counts.iteritems():
		if value < 3:
			to_remove.append(key)
	for key in to_remove:
		del feat_counts[key]

	feat_dict = {}
	i = 0
	for key in feat_counts.keys():
		feat_dict[key] = i
		i += 1

	nr_feat = len(feat_counts) 
	dataset = np.zeros((line_count, nr_feat+1), dtype=float)
	class_set = np.zeros([line_count,1], dtype=int)

	class_table = {}
	class_table["comp.windows.x"] = 0
	class_table["rec.autos"] = 1
	class_table["talk.politics.guns"] = 2

	with open(filename, 'r') as file:
		line_number = 0
		for line in file:
			tokens = line.split(" ")
			for feat in tokens[0:-1]:
				name, counts = feat.split(":")
				if name in feat_dict:
					dataset[line_number,feat_dict[name]] = int(counts)
			dataset[line_number,len(feat_counts)] = class_table[str(tokens[-1]).replace("#label#:","").replace("\n","")]
			line_number += 1

	return dataset

def create_model(training_set):

	model = {}

	feature_list_size = len(training_set[0]) - 1

	#Calculate prior probabilities
	class_variable_list = []
	classes = []
	for line in training_set:
		class_variable_list.append(line[-1])
	class_variable_count = {}
	prior_probabilities = {}
	for line in class_variable_list:
		if line not in class_variable_count.keys():
			class_variable_count[line] = 1
			if line not in classes:
				classes.append(line)
		else:
			class_variable_count[line] += 1
	for key in class_variable_count.keys():
		prior_probabilities[key] = float(class_variable_count[key]) / float(len(class_variable_list))
	
	#Calculate likelihood
	#1. Get feature count
	feature_count_per_column = {}
	for class_label in classes:
		if class_label not in feature_count_per_column.keys():
			feature_count_per_column[class_label] = []
		for i in range(feature_list_size):
			count = 0
			for j in range(len(training_set)):
				if training_set[j][-1] == class_label:
					count += training_set[j][i]
			feature_count_per_column[class_label].append(count)

	#2. Get total count per class
	total_count_per_class = {}
	for class_label in classes:
		if class_label not in total_count_per_class.keys():
			total_count_per_class[class_label] = 0
		for feature_count in feature_count_per_column[class_label]:
			total_count_per_class[class_label] += feature_count

	#3. Calculate likelihood for each feature given each class_label
	likelihood = {}
	for class_label in classes:
		if class_label not in likelihood.keys():
			likelihood[class_label] = []
		for feature_count in feature_count_per_column[class_label]:
			likelihood_probability = float(feature_count + smoothing_param) / float(total_count_per_class[class_label] + (feature_list_size*smoothing_param))
			likelihood[class_label].append(likelihood_probability)

	model["classes"] = classes
	model["class_variable_count"] = class_variable_count
	model["prior"] = prior_probabilities
	model["likelihood"] = likelihood

	return model

def predict(model,input_vector):

	predictions = {}
	classes = model["classes"]
	prior_probabilities = model["prior"]
	maximum_likelihood_list = model["likelihood"]

	input_vector = input_vector[0:-1]

	getcontext().prec = 100

	for class_label in classes:
		if class_label not in predictions.keys():
			predictions[class_label] = math.log(prior_probabilities[class_label])
		for i in range(len(input_vector)):
			predictions[class_label] += (input_vector[i]*math.log(maximum_likelihood_list[class_label][i]))



	result = predictions[0]
	classification = 0
	for class_label in classes:
		if predictions[class_label] > result:
			result = predictions[class_label]
			classification = int(class_label)


	return classification


def make_predictions(model,test_set):

	predictions_list = []
	for input_vector in test_set:
		predictions_list.append(predict(model,input_vector))

	return predictions_list

def calculate_accuracy(predictions_list,test_set,model):

	true_negatives = 0
	false_positives = 0
	false_negatives = 0

	success_count = 0
	for i in range(len(predictions_list)):
		if predictions_list[i] == test_set[i][-1]:
			success_count += 1

	accuracy = (float(success_count) / float(len(predictions_list))) * 100


	class_list = model["class_variable_count"]

	f_score = {}
	for class_label in class_list.keys():
		true_positives = 0
		class_count_input = 0
		class_count_output = 0
		if class_label not in f_score.keys():
			f_score[class_label] = []
		for i in range(len(test_set)):
			if class_label == test_set[i][-1]:
				class_count_input += 1
				if test_set[i][-1] == predictions_list[i]:
					true_positives += 1
			if class_label == predictions_list[i]:
				class_count_output += 1

		precision = float(true_positives) / float(class_count_output)
		recall = float(true_positives) / float(class_count_input)
		f_score[class_label].append(precision)
		f_score[class_label].append(recall)
		if float(precision + recall) == 0.0:
			f_score[class_label].append(0)
		else:
			f_score[class_label].append(float(2 * precision * recall) / float(precision + recall))

	macro_f_score = 0
	print "Class\tF-Score\t\tPrecision\tRecall"
	for class_label in f_score.keys():
		macro_f_score += f_score[class_label][0]
		print "%s\t%f\t%f\t%f"%(class_label,f_score[class_label][2],f_score[class_label][0],f_score[class_label][1])
	macro_f_score /= float(3)

	print "macro_f_score : ",macro_f_score
	print "accuracy : "+str(round(accuracy,2))+"%"

def main():

	start_time = time.time()
	current_path = os.getcwd()

	print "\nloading training data..."
	filename = current_path + "/dataset_preprocessing/train/training_set.txt"
	training_set = load_data(filename)
	print training_set

	#creating a model
	print "\ncreating model..."
	model = create_model(training_set)

	print "\nloading test data..."
	filename = current_path + "/dataset_preprocessing/test/test_set.txt"
	test_set = load_data(filename)
	print test_set

	#Make prediction
	print "\nmaking predictions..."
	predictions_list = make_predictions(model,test_set)

	#Calculate accuracy
	print "\ncalculating metrics...\n"
	calculate_accuracy(predictions_list,test_set,model)

	#Calculate runtime
	stop_time = time.time()

	print "runtime : "+str((stop_time - start_time) / 60)+" minutes!"

main()
