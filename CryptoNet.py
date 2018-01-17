from poloniex import polonitrade
import time, datetime, math
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

secret = '782c06450cacda6c91c5f6ea2842e7c70b1a86889721922c711c749dd86be9819ee3e1264fa6a3bfdc0f90660629ecef88182da3788e9cee0831cf1f1b8e32a8'
key = 'WRO9ZNTF-4H6TUE1Q-G50LJUF0-JH6JV4SH'

start_time = time.time()
def main():
	i_nodes = 360
	o_nodes = 180

	batch = createBatch()
	time0 = time.time()
	print('batch length:',len(batch))
	print('createBatch time:',time0 - start_time,'\n')

	data, labels = createDataAndLabels(batch, i_nodes, o_nodes)
	time2 = time.time()
	print('data and labels shape:', data.shape, labels.shape)
	print('createDataAndLabels time:',time2 - start_time,'\n')

	tfGraph(data,labels,i_nodes,o_nodes)


#create array filled with relevant info (oldest to newest)
def createBatch():
	pt = polonitrade.Poloniex()
	cP = 'USDT_BTC'
	batches = 1
	batch = []

	batch_time = time.time()
	for b in range(batches):
		next_batch = pt.marketTradeHist(currencyPair=cP,start=0,end=batch_time) #newest to oldest
		batch_time = int(time.mktime(datetime.datetime.strptime(next_batch[-1]['date'], "%Y-%m-%d %H:%M:%S").timetuple()))
		for n in range(len(next_batch)):
			transaction = next_batch[n]
			t_time = time.mktime(datetime.datetime.strptime(transaction['date'], "%Y-%m-%d %H:%M:%S").timetuple())
			total = float(transaction['total'])
			rate = float(transaction['rate'])
			order_type = transaction['type']
			if order_type == 'buy':
				order_type = 1.0
			else:
				order_type = -1.0
			t_list = [t_time, total, rate, order_type]
			batch.insert(0,t_list)
	return batch


def createDataAndLabels(batch, i_nodes, o_nodes):
	data_len = len(batch) - i_nodes
	data = np.empty([data_len, i_nodes])
	times_temp = np.empty([data_len])

	'''
	create data and matching time signatures
	'''
	for x in range(data_len):
		root_transaction = batch[x + i_nodes]
		times_temp[x] = root_transaction[0] #store root_transaction time
		rate_diff = np.empty([i_nodes]) #to store rate difference between root and previous <i_nodes> transactions
		for pt in range(i_nodes):
			branch_transaction = batch[x + pt]
			rate_diff = abs(root_transaction[2] - branch_transaction[2])
			munge = math.log1p(root_transaction[1] * rate_diff)
			if branch_transaction[3] < 0.0:
				munge *= -1.0
			data[x][pt] = munge

	temp_labels = {}
	temp = {}
	for t in batch:
		t_time = t[0]
		t_rate = t[2]
		if t_time in temp_labels:
			if t_time in temp:
				temp[t_time][0] += 1
				temp[t_time][1] += t_rate
			else:
				temp[t_time] = [1,t_rate]
		else:
			temp_labels[t_time] = t_rate
	for key in sorted(temp):
		occurence = temp[key][0]
		price_sum = temp[key][1]
		if key in temp_labels:
			occurence += 1
			price_sum += temp_labels[key]
		avg_rate = price_sum / occurence
		temp_labels[key] = avg_rate

	'''
	interpolate market values for actual time span
	store in dict as 'time interval: market price (interpolated)'
	'''
	x_points = []
	y_points = []
	for key in sorted(temp_labels):
		x_points.append(key)
		y_points.append(temp_labels[key])
	start_time, end_time = times_temp[0], times_temp[-1]
	time_range = np.arange(start_time, end_time)
	price_interpolation = np.interp(time_range, x_points, y_points)
	price_at_time = {}
	for x in range(price_interpolation.shape[0]):
		price_at_time[time_range[x]] = price_interpolation[x]

	'''store label in array with idexes corresponding to data array'''
	labels = np.empty([data_len, o_nodes])
	for fill_labels in range(data_len):
		for fill_nodes in range(o_nodes):
			x = float(fill_nodes)
			if times_temp[fill_labels] + x in price_at_time:
				labels[fill_labels][fill_nodes] = price_at_time[times_temp[fill_labels] + x]

	return data, labels


def tfGraph(data,labels,i_nodes,o_nodes):
	sess = tf.Session()
	x = tf.placeholder(tf.float32,shape=[None, i_nodes])
	y_ = tf.placeholder(tf.float32, shape=[None, o_nodes])

	W = tf.Variable(tf.zeros([i_nodes,o_nodes]))
	b = tf.Variable(tf.zeros([o_nodes]))

	sess.run(tf.global_variables_initializer())

	y = tf.matmul(x,W) + b

	cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

	batch_size = 1000
	training_slice = int((data.shape[0] / batch_size) * 3/4)
	for step in range(training_slice):
		train_data = data[step * batch_size:(step + 1) * batch_size]
		train_labels = labels[step * batch_size:(step + 1) * batch_size]
		train_step.run(session=sess,feed_dict={x: train_data, y_: train_labels})
		print('Training cycle:',step)

	test_data = data[training_slice:]
	test_labels = labels[training_slice:]
	print(test_data.shape, test_labels.shape)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval(session=sess,feed_dict={x: test_data, y_: test_labels}))


main()

end_time = time.time()
print('run time:',end_time - start_time)
