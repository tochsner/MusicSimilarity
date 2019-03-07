class Logger():
    def __init__(self, name):
        self.name = name

    def log(self, text):
        with open(self.name + ".txt", "a+") as log_file:
            log_file.write(str(text) + "\n")
