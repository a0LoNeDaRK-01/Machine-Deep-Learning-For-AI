# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:33:06 2023

@author: Coder
"""

import json

data = '{"firstName":"Enes","lastName":"Yalcin"}'

y = json.loads(data)

print(y["firstName"])
print(y["lastName"])

customer = {
                
             "firstName":"Enes",
             "lastName":"Yalcin"
    }

customerJson = json.dumps(customer)

print(customer)

print(json.dumps("Mert"))