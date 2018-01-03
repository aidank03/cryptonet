from poloniex import polonitrade
import time, datetime, math
import tensorflow as tf
import numpy as np

start_time = time.time()

secret = '782c06450cacda6c91c5f6ea2842e7c70b1a86889721922c711c749dd86be9819ee3e1264fa6a3bfdc0f90660629ecef88182da3788e9cee0831cf1f1b8e32a8'
key = 'WRO9ZNTF-4H6TUE1Q-G50LJUF0-JH6JV4SH'



def main():
	batch = createBatch()
	time0 = time.time()
	print('batch = [time, total, rate, type]')
	print(batch[10:])
	print('batch shape:',batch.shape)
	print('createBatch time:',time0 - start_time,'\n')


	o_nodes = 30
	labels = createLabels(batch,o_nodes)
	time1 = time.time()
	print('labels length:',len(labels))
	print('createLabels time:',time1 - start_time)


	i_nodes = 180
	training_data = createTrainingData(batch, labels, i_nodes)
	time2 = time.time()
	print(training_data[:5])
	print('training_data shape:',training_data.shape)
	print('createTrainingData time:',time2 - start_time,'\n')
	

	'''tfGraph(training_data,labels,i_nodes,o_nodes)'''


# create array filled with relevant info (oldest to newest)
def createBatch():
	pt = polonitrade.Poloniex()
	cP = 'USDT_BTC'
	batches = 1
	batch = np.empty([50000 * batches,4])

	batch_time = time.time()
	for b in range(batches):
		next_batch = pt.marketTradeHist(currencyPair=cP,start=0,end=batch_time)
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
			reverse = -1 - ((b * 50000) + n)
			batch[reverse][0] = t_time
			batch[reverse][1] = total
			batch[reverse][2] = rate
			batch[reverse][3] = order_type
	return batch


def createLabels(batch,o_nodes):
	labels = {}
	temp = {}
	for t in batch:
		t_time = t[0]
		t_rate = t[2]
		if t_time in labels:
			if t_time in temp:
				temp[t_time][0] += 1
				temp[t_time][1] += t_rate
			else:
				temp[t_time] = [1,t_rate]
		else:
			labels[t_time] = t_rate
	for key in sorted(temp):
		avg_rate_from_temp = temp[key][1] / (temp[key][0])
		labels[key] = avg_rate_from_temp
	return labels


def createTrainingData(batch, labels, i_nodes):
	batch_length = batch.shape[0]
	training_data = np.empty([batch_length,i_nodes + 1])

	for current_t in range(batch_length - i_nodes):
		training_data[current_t][0] = batch[current_t + i_nodes][0]
		rate_diff = np.empty([i_nodes])
		transaction = batch[current_t + i_nodes]
		log_total = math.log1p(transaction[1])
		for previous_t in range(i_nodes):
			p = batch[current_t - previous_t]
			log_rate_diff = math.log1p(abs(transaction[2] - p[2]))
			munge = math.log1p(log_total * log_rate_diff)
			if p[3] < 0.0:
				munge *= -1.0
			training_data[current_t][previous_t + 1] = munge
	return training_data

# not complete
def tfGraph(training_data,labels,i_nodes,o_nodes):
	sess = tf.Session()
	x = tf.placeholder(tf.float32,shape=[None, i_nodes])
	y_ = tf.placeholder(tf.float32, shape=[None, o_nodes])

	W = tf.Variable(tf.zeros([i_nodes,o_nodes]))
	b = tf.Variable(tf.zeros([o_nodes]))

	sess.run(tf.global_variables_initializer())

	y = tf.matmul(x,W) + b

	cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


	for cycle in range(100):
  		train_batch = training_data[cycle * 100:(cycle+1) *100][1:]
		label_batch = labels[training_data[cycle * 100:(cycle+1) *100][0]]
  		train_step.run(feed_dict={x: train_batch, y_: label_batch})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




main()

end_time = time.time()
print('run time:',end_time - start_time)
