from Drain import LogParser
import os

input_dir = 'C:\\Users\\chris\\OneDrive\\Desktop\\College\\ARDC-Research'
logfile = 'train_logfile.log'
output_dir = 'parsed_train_logfile'

log_format='<Date> <Time> <Pid> <Level> <Component>: <Content>'

regex = []

parser = LogParser(log_format=log_format,indir=input_dir,outdir=output_dir,depth=4,st=0.5,rex=regex)

parser.parse(logfile)