from poloniex import polonitrade
import time, datetime, math
import tensorflow as tf
import numpy as np

time0 = time.time()

secret = '782c06450cacda6c91c5f6ea2842e7c70b1a86889721922c711c749dd86be9819ee3e1264fa6a3bfdc0f90660629ecef88182da3788e9cee0831cf1f1b8e32a8'
key = 'WRO9ZNTF-4H6TUE1Q-G50LJUF0-JH6JV4SH'



def main():
	batch = createBatch()
	print(batch[0][0],batch[-1][0])

	input_nodes = 300
	output_nodes = 20
	labels = np.empty(1)


# create array filled with relevant info (oldest to newest)
def createBatch():
	pt = polonitrade.Poloniex()
	cP = 'USDT_BTC'
	batches = 2
	batch = np.empty([50000 * batches])

	batch_time = time.time()
	for b in range(batches):
		next_batch = pt.marketTradeHist(currencyPair=cP,start=0,end=batch_time)
		batch_time = int(time.mktime(datetime.datetime.strptime(next_batch[-1]['date'], "%Y-%m-%d %H:%M:%S").timetuple()))
		for n in next_batch:
			transaction = next_batch[-1 - n]
			t_time = time.mktime(datetime.datetime.strptime(transaction['date'], "%Y-%m-%d %H:%M:%S").timetuple())
			total = float(transaction['total'])
			price = float(transaction['rate'])
			order_type = transaction['type']
			if order_type == 'buy':
				order_type = -1.0
			else:
				order_type = 1.0
			info = np.array([t_time,total,price,order_type])
			batch[-1 - ((b * 50000) + n)] = info

	return batch


'''

# reverse order to account for newest order being first in batch

# prepare training_data and labels
	start_time = batch_time
	time_diff =

	for t in range(len(batch) - input_nodes):
		transaction_time = int(time.mktime(datetime.datetime.strptime(batch[t]['date'], "%Y-%m-%d %H:%M:%S").timetuple()))
		for x in range(output_nodes):


		t_current = batch[t]
		total = math.log1p(float(current['total']))
		price = float(current['rate'])
		price_diff = []
		for n in range(input_nodes):
			n_current = batch[t + n]
			n_next_price = float(n_current['rate'])
			price_diff_of_next = abs(price - n_next_price)
			munge = math.log1p(total * math.log1p(price_diff_of_next))
			price_diff.append(munge)
		training_data.append(price_diff)
	print(len(training_data))

# prepare labels

for el in training_data:
	price = float(el['rate'])
	runningMarketValue += price
	orderType = el['type']
	if orderType == 'buy':
		order = {'amount': el['amount'], 'price': price}
		print('Buy',order['amount'],cP,'@',order['price'])
		buyList.append(order)
	if orderType == 'sell':
		order = {'amount': el['amount'], 'price': price}
		print('Sell',order['amount'],cP,'@',order['price'])
		sellList.append(order)

print(len(training_data))
print(training_data[0],training_data[-1])
print('Market:',runningMarketValue,'USD')


sess = tf.Session()
x = tf.placeholder(tf.float32,[None, ])

print('Buy:',*buyList, sep='\n')
print('\n')
print('Sell:',*sellList, sep='\n')

t1 = time.time() - t0
time.sleep(60 - t1)
'''

main()

time1 = time.time()
print(time1 - time0)
