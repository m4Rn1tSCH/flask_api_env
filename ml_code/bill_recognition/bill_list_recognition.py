import os
import pandas as pd
#from collections import Counter

link = r"C:\Users\bill-\Desktop\*.csv"
new_link = link.replace(os.sep, '/')
input_dir = ''.join(('', new_link,''))

#CSV verifier
#def is_csv(infile):
#	try:
#		with open(infile, newline='') as csvfile:
#			start = csvfile.read(4096)

# isprintable does not allow newlines, printable does not allow umlaute...
#			if not all([c in string.printable or c.isprintable() for c in start]):
#				return False
#			dialect = csv.Sniffer().sniff(start)
#			else:
#			return True
#	except csv.Error:
#Could not get a csv dialect -> probably not a csv.
#	return Fals

#read csv or excel
if input.is_csv == True:
	input = pd.read_csv(input_dir)
else:
	input = pd.read_excel(input_dir)

############################################
#find columns first
bills_found = []

try:
	column_list = input.items()
	if column_list.str(bill_list):
		for n in bill_list:
		bills_found = bills_found.append(bill_list)
	else:
		print("Sorry, no known bills recognized :*(")
except:
#specify input manually
	
#initiate row iterator
##row = next(input.iterrows())[input.get([T,t]ransactions,[S,s]hopname]
##row = input.iterrows()
#for label, content in row:
##	print('label:', label , sep = '\n')
##	print('content:', content , sep = '\n')

for label, content in input.items():
		print('label:', label)
		print('content:', content, sep='\n')


