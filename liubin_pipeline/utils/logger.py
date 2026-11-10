import time

def logging(args, content):
    with open(args.logging_path, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\t' + str(content) + '\n')