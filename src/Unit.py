from enum import Enum
import re
from .log import *


class BasicUnitLength(Enum):
    mm = 1
    cm = 10
    dm = 100
    m = 1000
    km = 10000
class BasicUnitMass(Enum):
    g = 1
    kg = 1000
    t = 10000000
class BasicUnitTime(Enum):
    s = 1
    min = 60
    h = 3600

# * 用户输入的单位基准，整个模块使用的基准也是 mm、g、s 且不可改变
baseLengthUnit = BasicUnitLength.mm
baseMassUnit = BasicUnitMass.g
baseTimeUnit = BasicUnitTime.s

# @classmethod
# def CoverToSysUnitDec(cls, func):
#     def wapper(*args, **kwargs):
#         x = func(*args, **kwargs)

#         return x

#     return wapper

# @staticmethod
def ConvertToBaseUnit(value:float, unit:str):
    input = unit
    # re.sub(r'\*', '-*-', exp)
    # re.sub(r'\\', '-|-', exp)
    unit = re.sub(r'\b(mm|MM|Mm|mM)\b', str(BasicUnitLength.mm.value/baseLengthUnit.value), unit)
    unit = re.sub(r'\b(cm|CM|Cm|cM)\b', str(BasicUnitLength.cm.value/baseLengthUnit.value), unit)
    unit = re.sub(r'\b(dm|DM|Dm|dM)\b', str(BasicUnitLength.dm.value/baseLengthUnit.value), unit)
    unit = re.sub(r'\b(m|M)\b', str(BasicUnitLength.m.value/baseLengthUnit.value), unit)
    unit = re.sub(r'\b(km|KM|kM|Km)\b', str(BasicUnitLength.km.value/baseLengthUnit.value), unit)

    unit = re.sub(r'\b(g|G)\b', str(BasicUnitMass.g.value/baseMassUnit.value), unit)
    unit = re.sub(r'\b(kg|KG|kG|Kg)\b', str(BasicUnitMass.kg.value/baseMassUnit.value), unit)
    unit = re.sub(r'\b(t|T)\b', str(BasicUnitMass.t.value/baseMassUnit.value), unit)


    unit = re.sub(r'\b(s|S|SEC|sec|Sec)\b', str(BasicUnitTime.s.value/baseTimeUnit.value), unit)
    unit = re.sub(r'\b(min|MIN|Min)\b', str(BasicUnitTime.min.value/baseTimeUnit.value), unit)
    unit = re.sub(r'\b(h|H|Hour|hour|hr|Hr)\b', str(BasicUnitTime.h.value/baseTimeUnit.value), unit)

    x = re.findall(r'[a-zA-Z]', unit)
    if x:
        logger.fatal('Unit:{} can not be resolved'.format(x))

        raise Exception('Unit:{} can not be resolved'.format(x))
    else:
        logger.info('Unit:{} has been resolved to {}'.format(input, unit))
    
    return value * eval(unit)