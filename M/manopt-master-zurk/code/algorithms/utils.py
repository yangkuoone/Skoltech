from time import gmtime, strftime


def time():
    return strftime("%H:%M:%S ", gmtime())
