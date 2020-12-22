#!/usr/bin/python3


def dictIterate(ddict:dict, max_width:int = 120,curr_width:int = 0):
    msg = ""
    for key, val in ddict.items():
        if type(val) is dict:
            msg="{}\n{}: ".format(msg,key)
            msg1 ="{" + dictIterate(val,  curr_width=len(key)+2) + "}"
            msg = "{}{}\n".format(msg,msg1)
            curr_width=0
            continue

        msg1 = "{}:{}".format(key, val)
        if curr_width + len(msg1) > max_width:
            msg = "\n{}".format(msg)
            curr_width = 0
        curr_width += len(msg1)
        msg = "{} {}".format(msg, msg1)
    # msg="{}\n".format(msg)
    return msg

def listIterate(llist:list,max_width:int = 120,curr_width:int = 0):
    msg="["
    for item in llist:
        msg1 = "{},".format(item)
        if curr_width + len(msg1) > max_width:
            msg = "\n{}".format(msg)
            curr_width = 0
        curr_width += len(msg1)
        msg = "{} {}".format(msg, msg1)
    # msg = "{}\n]".format(msg)
    msg = "{}]".format(msg)
    return msg