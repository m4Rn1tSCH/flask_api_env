import sys
import json

#specify keys
KEYS = ['acquire_campaign', 'acquire_source', 'advertising_id', 'android_id', 'app', 'app_ver', 'at', 'birth_date', 'browser', 'carrier', 'city_name', 'closed_at', 'clv_change', 'clv_total', 'country', 'custom', 'custom_X', 'customer_ids', 'device_new', 'device_timezone', 'device_uuid', 'install_id', 'language', 'lat', 'library_ver', 'lng', 'marketing_campaign', 'marketing_medium', 'marketing_source', 'model', 'name', 'nth', 'os_ver', 'platform', 'report_attr', 'screen_flow', 'sec_since_last_session', 'session_uuid', 'total_session_sec', 'type', 'user_type', 'uuid']
#manual file input
f = input("File Path: ")
op = open(f.split(".")[0]+".csv","w")

#PREPARING HEADER FOR CSV DOCUMENT
HEADER = ""
j = 0
for k in KEYS:
	HEADER += k.upper()
	if(j < len(KEYS)):
		HEADER += ","
	j += 1

op.write(HEADER+"\n")
with open(f) as file:
	content = file.readlines()
	for line in content:
		try:
			data = json.loads(line)
			line_data = ""
			j = 0
			for k in KEYS:
				if k in data:
					line_data += '"'+str(data[k]).replace('"','\"')+'"'
				else:
					line_data += " "
				if(j < len(KEYS)):
					line_data += ","
				j += 1
			
			op.write(line_data+"\n")
		except Exception as inst:
			print(type(inst))
			print(inst.args)
			print(inst)
op.close()