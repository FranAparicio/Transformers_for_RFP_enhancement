import re
import sys
import matplotlib.pyplot as plt

# Define regular expression pattern to extract values
# For example: {'loss': 4.8205, 'learning_rate': 4.7619047619047613e-08, 'epoch': 2.86}
pattern_train = r"'loss': ([\d\.]+),.*'epoch': ([\d\.]+)"
pattern_eval = r"'eval_loss': ([\d\.]+)"

# Set the first argument given as the input file
if len(sys.argv) < 2:
	print("Usage: python script.py input_file")
	sys.exit(1)
input_file = sys.argv[1]

# Open file and extract values
with open(input_file,'r') as f:
	lines = f.readlines()
	train_loss = []
	eval_loss = []
	epoch = []
	for line in lines:
		match_train = re.search(pattern_train,line)
		match_eval = re.search(pattern_eval,line)
		if match_train:
			train_loss.append(float(match_train.group(1)))
			epoch.append(float(match_train.group(2)))
		if match_eval:
			eval_loss.append(float(match_eval.group(1)))
	print(epoch,eval_loss,train_loss)

# Plot the line for the training
fig, ax = plt.subplots()
ax.scatter(epoch,train_loss,label="Training loss",color='green',s=5)
ax.scatter(epoch,eval_loss,label="Validation loss",color='blue',s=5)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.show()
