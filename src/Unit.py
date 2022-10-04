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
UserLengthUnit = BasicUnitLength.mm
UserMassUnit = BasicUnitMass.g
UserTimeUnit = BasicUnitTime.s

# @classmethod
# def CoverToSysUnitDec(cls, func):
#     def wapper(*args, **kwargs):
#         x = func(*args, **kwargs)

#         return x

#     return wapper

# @staticmethod
def ConvertToBaseUnit(value:float, unit:str):
    input = unit

    # * Pressure Pa
    unit = re.sub(r'\b(mpa|MPA|mPa|mPA|Mpa|MPa)\b', '1000000*N/m/m', unit)
    unit = re.sub(r'\b(pa|PA|Pa)\b', 'N/m/m', unit)

    # * Force N
    unit = re.sub(r'\b(kN|kn|Kn|KN)\b', '1000*kg*m*m/s', unit)
    unit = re.sub(r'\b(N|n)\b', 'kg*m*m/s', unit)
    
    #* basicUnit-Length
    unit = re.sub(r'\b(mm|MM|Mm|mM)\b', str(BasicUnitLength.mm.value/UserLengthUnit.value), unit)
    unit = re.sub(r'\b(cm|CM|Cm|cM)\b', str(BasicUnitLength.cm.value/UserLengthUnit.value), unit)
    unit = re.sub(r'\b(dm|DM|Dm|dM)\b', str(BasicUnitLength.dm.value/UserLengthUnit.value), unit)
    unit = re.sub(r'\b(m|M)\b', str(BasicUnitLength.m.value/UserLengthUnit.value), unit)
    unit = re.sub(r'\b(km|KM|kM|Km)\b', str(BasicUnitLength.km.value/UserLengthUnit.value), unit)

    #* basicUnit-Mass
    unit = re.sub(r'\b(g|G)\b', str(BasicUnitMass.g.value/UserMassUnit.value), unit)
    unit = re.sub(r'\b(kg|KG|kG|Kg)\b', str(BasicUnitMass.kg.value/UserMassUnit.value), unit)
    unit = re.sub(r'\b(t|T)\b', str(BasicUnitMass.t.value/UserMassUnit.value), unit)

    #* basicUnit-Time
    unit = re.sub(r'\b(s|S|SEC|sec|Sec)\b', str(BasicUnitTime.s.value/UserTimeUnit.value), unit)
    unit = re.sub(r'\b(min|MIN|Min)\b', str(BasicUnitTime.min.value/UserTimeUnit.value), unit)
    unit = re.sub(r'\b(h|H|Hour|hour|hr|Hr)\b', str(BasicUnitTime.h.value/UserTimeUnit.value), unit)

    # *是否有未解析的单位
    x = re.findall(r'\b[a-zA-Z]+\b', unit)
    if x:
        StandardLogger.fatal('Unit:{} can not be resolved'.format(x))

        raise Exception('Unit:{} can not be resolved'.format(x))
    else:
        StandardLogger.info('Unit:{} has been resolved to {}'.format(input, unit))
    
    return value * eval(unit)