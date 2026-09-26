from agents.bc import BCAgent
from agents.cdp import CDPAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.qam import QAMAgent
from agents.vgf import VGFAgent

agents = {
    'bc': BCAgent,
    'cdp': CDPAgent,
    'fbrac': FBRACAgent,
    'fql': FQLAgent,
    'ifql': IFQLAgent,
    'iql': IQLAgent,
    'rebrac': ReBRACAgent,
    'qam': QAMAgent,
    'vgf': VGFAgent
}
